using System.Net;
using UnityEngine;

public class BallMusicVisualizer : MonoBehaviour
{
    [Header("References")]
    public YinPitchTracker pitchTracker;
    public AudioSource audioSource;
    public MeshRenderer ballRenderer;

    [Header("Pitch to Y-Axis Settings")]
    public float minY = 0f;
    public float maxY = 10f;
    public float minFreq = 40f;
    public float maxFreq = 2000f;

    [Header("Amplitude Visual Settings")]
    [ColorUsage(true, true)]
    public Color baseColor = Color.white;

    private YinPitchTracker.FrameAnnotation[] analysisData;
    private Material ballMaterial;
    private int currentFrameIndex = 0;

    // ✅ Precomputed data
    private float[] precomputedY;
    private Color[] precomputedColor;
    private Color[] precomputedEmission;

    void Start()
    {
        ballMaterial = ballRenderer.material;
        ballMaterial.EnableKeyword("_EMISSION");

        if (pitchTracker != null && audioSource != null && audioSource.clip != null)
        {
            analysisData = pitchTracker.AnalyzeClip(
                audioSource.clip,
                0,
                new int[] { 0, 2, 4, 5, 7, 9, 11 },
                40f,
                2000f,
                2048,
                1024,
                0.1f,
                0.6f
            );

            PrecomputeFrames(); // 🚀 Important
        }
    }


    // DIFFERENCES -- changed computation of Y position from update to start for less lag
    void PrecomputeFrames()
    {
        int length = analysisData.Length;
        float smoothingFactor = 0.1f; // Adjust for more or less smoothing

        precomputedY = new float[length];
        precomputedColor = new Color[length];
        precomputedEmission = new Color[length];

        float logDenominator = Mathf.Log(maxFreq / minFreq, 2f);
        float lastValidY = transform.position.y;
        // DIFFERENCES -- added buffering logic to handle sudden drops in frequency and prevent jittery movement
        int fBuffer = 0;
        float lastFreq = analysisData[0].frequency_hz;
        for (int i = 0; i < length; i++)
        {
            var frame = analysisData[i];
            float freq = frame.frequency_hz;

            if (frame.midi == -1 || frame.frequency_hz <= minFreq)
            {
                precomputedY[i] = lastValidY; // 👈 smooth fallback
            }
            else if ((freq < lastFreq - 1 || freq > lastFreq + 1) && fBuffer < 10) // --- Buffering logic to handle sudden drops ---
            {
                precomputedY[i] = lastValidY; // 👈 hold previous value
                fBuffer++;
            } else 
            {

                float logFreq = Mathf.Log(frame.frequency_hz / minFreq, 2f) / logDenominator;
                float normalized = Mathf.Clamp01(logFreq);
                float y = Mathf.Lerp(minY, maxY, normalized);
                Debug.Log("Y position: " + frame.frequency_hz);

                precomputedY[i] = y;
                lastValidY = y; // 👈 update
                fBuffer = 0; // 👈 reset buffer
                lastFreq = freq;
            }
            float t = Mathf.Clamp01(frame.rms * 10f);
            Color surface = Color.Lerp(baseColor, Color.white, t);

            precomputedColor[i] = surface;
            precomputedEmission[i] = surface * t;
        }
        // DIFFERENCES -- replaced smoothing in the code
        for (int i = 1; i < length; i++)
        {
            precomputedY[i] = Mathf.Lerp(precomputedY[i - 1], precomputedY[i], smoothingFactor);
        }
    }

    void Update()
    {
        if (analysisData == null || analysisData.Length == 0 || !audioSource.isPlaying)
            return;

        float currentTime = audioSource.time;
        int frameIndex = GetFrameIndexAtTime(currentTime);

        if (frameIndex < 0)
            return;

        ApplyPrecomputedFrame(frameIndex);
    }

    void ApplyPrecomputedFrame(int index)
    {
        // --- Position (no math, just assignment) ---
        Vector3 pos = transform.position;
        pos.y = precomputedY[index];
        transform.position = pos;

        // --- Visuals (no math) ---
        if (ballMaterial.HasProperty("_BaseColor"))
            ballMaterial.SetColor("_BaseColor", precomputedColor[index]);
        else if (ballMaterial.HasProperty("_Color"))
            ballMaterial.SetColor("_Color", precomputedColor[index]);

        if (ballMaterial.HasProperty("_EmissionColor"))
            ballMaterial.SetColor("_EmissionColor", precomputedEmission[index]);
    }

    int GetFrameIndexAtTime(float time)
    {
        if (analysisData == null || analysisData.Length == 0)
            return -1;

        if (currentFrameIndex >= analysisData.Length || time < analysisData[currentFrameIndex].time_s)
            currentFrameIndex = 0;

        while (currentFrameIndex < analysisData.Length - 1 &&
               analysisData[currentFrameIndex + 1].time_s <= time)
        {
            currentFrameIndex++;
        }

        return currentFrameIndex;
    }
}
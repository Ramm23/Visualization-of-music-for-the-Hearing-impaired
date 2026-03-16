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
    public float smoothSpeed = 10f;

    [Header("Amplitude Visual Settings")]
    [ColorUsage(true, true)]
    public Color baseColor = Color.white;
    public float amplitudeScale = 25f;
    public float minAlpha = 0.2f;
    public float maxAlpha = 1.0f;
    public float amplitudeSmoothSpeed = 12f;

    private YinPitchTracker.FrameAnnotation[] analysisData;
    private Material ballMaterial;
    private int currentFrameIndex = 0;
    private float smoothedAmplitude = 0f;

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

        var frame = analysisData[frameIndex];

        UpdateBallPosition(frame);
        UpdateBallVisuals(frame);
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

    void UpdateBallPosition(YinPitchTracker.FrameAnnotation frame)
    {
        if (frame.midi == -1 || frame.frequency_hz <= 0f)
            return;

        float logFreq = Mathf.Log(frame.frequency_hz / minFreq, 2f) /
                        Mathf.Log(maxFreq / minFreq, 2f);

        float targetY = Mathf.Lerp(minY, maxY, Mathf.Clamp01(logFreq));

        Vector3 newPos = transform.position;
        newPos.y = Mathf.Lerp(newPos.y, targetY, Time.deltaTime * smoothSpeed);
        transform.position = newPos;
    }

    void UpdateBallVisuals(YinPitchTracker.FrameAnnotation frame)
    {
        float targetAmplitude = Mathf.Clamp01(frame.rms * amplitudeScale);
        smoothedAmplitude = Mathf.Lerp(
            smoothedAmplitude,
            targetAmplitude,
            Time.deltaTime * amplitudeSmoothSpeed
        );

        // Emission / brightness
        Color emissionColor = baseColor * smoothedAmplitude;
        if (ballMaterial.HasProperty("_EmissionColor"))
        {
            ballMaterial.SetColor("_EmissionColor", emissionColor);
            DynamicGI.SetEmissive(ballRenderer, emissionColor);
        }

        // Transparency
        float alpha = Mathf.Lerp(minAlpha, maxAlpha, smoothedAmplitude);
        Color surfaceColor = baseColor;
        surfaceColor.a = alpha;

        if (ballMaterial.HasProperty("_BaseColor"))
            ballMaterial.SetColor("_BaseColor", surfaceColor);
        else if (ballMaterial.HasProperty("_Color"))
            ballMaterial.SetColor("_Color", surfaceColor);
    }
}
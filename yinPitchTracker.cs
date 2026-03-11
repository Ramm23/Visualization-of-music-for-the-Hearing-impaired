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
    public float minFreq = 40f;   // e.g., Low E (E1)
    public float maxFreq = 2000f; // e.g., High B (B6)
    public float smoothSpeed = 10f;

    [Header("Amplitude to Brightness Settings")]
    [ColorUsage(true, true)] // Enables HDR for "Glowing" effect
    public Color baseColor = Color.white;
    public float sensitivity = 50f;

    private YinPitchTracker.FrameAnnotation[] analysisData;
    private Material ballMaterial;

    void Start()
    {
        ballMaterial = ballRenderer.material;

        if (pitchTracker != null && audioSource.clip != null)
        {
            // Trigger the C processing we built
            // analysisData = pitchTracker.AnalyzeClip(audioSource.clip); 
            // Note: Ensure your AnalyzeClip returns the array!
        }
    }

    void Update()
    {
        if (analysisData == null || !audioSource.isPlaying) return;

        float currentTime = audioSource.time;
        UpdateBallPosition(currentTime);
        UpdateBallBrightness();
    }

    void UpdateBallPosition(float time)
    {
        // Find the frame closest to the current audio playback time
        // In a production build, you'd use a binary search for performance
        foreach (var frame in analysisData)
        {
            if (frame.time_s >= time)
            {
                if (frame.midi != -1) // If a note is actually detected
                {
                    // Map frequency logarithmically to 0.0 - 1.0 range
                    float logFreq = Mathf.Log(frame.frequency_hz / minFreq, 2) / Mathf.Log(maxFreq / minFreq, 2);
                    float targetY = Mathf.Lerp(minY, maxY, Mathf.Clamp01(logFreq));

                    // Smooth the movement so it doesn't "teleport"
                    Vector3 newPos = transform.position;
                    newPos.y = Mathf.Lerp(newPos.y, targetY, Time.deltaTime * smoothSpeed);
                    transform.position = newPos;
                }
                break;
            }
        }
    }

    void UpdateBallBrightness()
    {
        // Placeholder Amplitude Algorithm: RMS (Root Mean Square)
        // This calculates the current volume of the audio playing right now
        float[] samples = new float[256];
        audioSource.GetOutputData(samples, 0); // Get current wave snapshot
        
        float sum = 0;
        for (int i = 0; i < samples.Length; i++) {
            sum += samples[i] * samples[i];
        }
        float rms = Mathf.Sqrt(sum / samples.Length); // This is your "Amplitude"

        // Map amplitude to Emission color (Brightness)
        float emission = rms * sensitivity;
        Color finalColor = baseColor * Mathf.LinearToGammaSpace(emission);
        
        ballMaterial.SetColor("_EmissionColor", finalColor);
        
        // Ensure the material is actually glowing (standard shader)
        DynamicGI.SetEmissive(ballRenderer, finalColor);
    }
}
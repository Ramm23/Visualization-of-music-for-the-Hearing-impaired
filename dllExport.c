#if defined(_WIN32)
#define DllExport __declspec(dllexport)
#else
#define DllExport
#endif

// Update your main processing function signature to include DllExport:
DllExport void process_audio_frames(...) { /* existing code */ }

// Add this new function to free the memory from the C# side:
DllExport void free_annotations(FrameAnnotation *annotations)
{
    if (annotations != NULL)
    {
        free(annotations);
    }
}
using UnityEngine;
using System.Collections.Generic;

/// <summary>
/// Navigate through LEGO assembly steps in Unity.
/// Attach this to an empty GameObject in your scene.
/// </summary>
public class AssemblyStepNavigator : MonoBehaviour
{
    [Header("Assembly Step Models")]
    [Tooltip("Drag all 7 step GameObject models here in order (step1 to step7)")]
    public List<GameObject> assemblySteps = new List<GameObject>();

    [Header("Settings")]
    [SerializeField]
    private int currentStep = 0;

    [SerializeField]
    private bool showInstructions = true;

    private void Start()
    {
        // Validate setup
        if (assemblySteps.Count == 0)
        {
            Debug.LogError("No assembly steps assigned! Please drag the step models into the inspector.");
            return;
        }

        // Show only the first step initially
        ShowStep(currentStep);

        if (showInstructions)
        {
            Debug.Log("=== LEGO Assembly Navigator ===");
            Debug.Log("Controls:");
            Debug.Log("  Right Arrow / D: Next step");
            Debug.Log("  Left Arrow / A: Previous step");
            Debug.Log("  Number Keys 1-7: Jump to specific step");
            Debug.Log("  Space: Toggle all steps on/off");
            Debug.Log("  H: Hide these instructions");
        }
    }

    private void Update()
    {
        // Navigation controls
        if (Input.GetKeyDown(KeyCode.RightArrow) || Input.GetKeyDown(KeyCode.D))
        {
            NextStep();
        }
        else if (Input.GetKeyDown(KeyCode.LeftArrow) || Input.GetKeyDown(KeyCode.A))
        {
            PreviousStep();
        }
        else if (Input.GetKeyDown(KeyCode.Space))
        {
            ToggleAllSteps();
        }
        else if (Input.GetKeyDown(KeyCode.H))
        {
            showInstructions = false;
        }

        // Direct step selection (1-7)
        for (int i = 1; i <= 7; i++)
        {
            if (Input.GetKeyDown(KeyCode.Alpha0 + i) && i <= assemblySteps.Count)
            {
                ShowStep(i - 1);
            }
        }
    }

    private void NextStep()
    {
        if (assemblySteps.Count == 0) return;

        currentStep = (currentStep + 1) % assemblySteps.Count;
        ShowStep(currentStep);
    }

    private void PreviousStep()
    {
        if (assemblySteps.Count == 0) return;

        currentStep--;
        if (currentStep < 0) currentStep = assemblySteps.Count - 1;
        ShowStep(currentStep);
    }

    private void ShowStep(int stepIndex)
    {
        if (stepIndex < 0 || stepIndex >= assemblySteps.Count) return;

        // Hide all steps
        foreach (var step in assemblySteps)
        {
            if (step != null)
                step.SetActive(false);
        }

        // Show selected step
        if (assemblySteps[stepIndex] != null)
        {
            assemblySteps[stepIndex].SetActive(true);
            currentStep = stepIndex;
            Debug.Log($"Showing Step {stepIndex + 1}/{assemblySteps.Count}");
        }
    }

    private void ToggleAllSteps()
    {
        bool anyActive = false;
        foreach (var step in assemblySteps)
        {
            if (step != null && step.activeSelf)
            {
                anyActive = true;
                break;
            }
        }

        // If any are active, hide all. Otherwise show current step.
        if (anyActive)
        {
            foreach (var step in assemblySteps)
            {
                if (step != null)
                    step.SetActive(false);
            }
            Debug.Log("All steps hidden");
        }
        else
        {
            ShowStep(currentStep);
        }
    }

    // GUI for on-screen display
    private void OnGUI()
    {
        if (!showInstructions || assemblySteps.Count == 0) return;

        GUIStyle style = new GUIStyle(GUI.skin.label);
        style.fontSize = 16;
        style.normal.textColor = Color.white;
        style.padding = new RectOffset(10, 10, 10, 10);

        string info = $"Step {currentStep + 1}/{assemblySteps.Count}\n\n";
        info += "Controls:\n";
        info += "  ← → or A/D: Navigate steps\n";
        info += "  1-7: Jump to step\n";
        info += "  Space: Toggle visibility\n";
        info += "  H: Hide this panel";

        GUI.Label(new Rect(10, 10, 300, 150), info, style);
    }
}

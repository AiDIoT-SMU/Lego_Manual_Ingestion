# Unity3D LEGO Assembly Visualizer Setup Guide

This guide will help you set up Unity3D to visualize your LEGO assembly steps.

## Prerequisites

- **Unity Hub**: Download from https://unity.com/download
- **Unity Editor**: Version 2022.3 LTS or newer (installed via Unity Hub)
- **Operating System**: macOS, Windows, or Linux

## Step 1: Install Unity

1. **Download Unity Hub**
   - Go to https://unity.com/download
   - Download and install Unity Hub for your OS

2. **Install Unity Editor**
   - Open Unity Hub
   - Go to "Installs" tab
   - Click "Install Editor"
   - Select **Unity 2022.3 LTS** (Long Term Support)
   - Click "Next" and follow the installation wizard
   - No extra modules required for basic visualization

## Step 2: Create a New Unity Project

1. **Create Project**
   - Open Unity Hub
   - Click "New Project" button
   - Select **"3D Core"** template
   - Project Name: `LEGO_Assembly_Visualizer`
   - Location: Choose where to save (e.g., Desktop)
   - Click "Create Project"

2. **Wait for Unity to Initialize**
   - First launch takes 1-2 minutes
   - Unity will open with an empty scene

## Step 3: Import Your LEGO Models

1. **Locate Your Generated Models**
   - Your .obj files are in: `/Users/jay/Desktop/CS480/Lego_Assembler2/data/processed/123456/unity_models/`
   - You should have 7 files: `sg50_step1.obj` through `sg50_step7.obj`

2. **Import into Unity**
   - In Unity, find the "Project" panel (bottom of screen)
   - Right-click in the Assets folder
   - Select "Create" → "Folder", name it "Models"
   - Open Finder/Explorer and navigate to your .obj files
   - **Drag all 7 .obj files** into the Unity "Models" folder
   - Unity will automatically import them

3. **Configure Import Settings (Optional)**
   - Click on any imported .obj file
   - In the Inspector (right panel), check:
     - Scale Factor: 1 (if models are too small/large, adjust this)
     - Generate Colliders: Off (not needed for visualization)
   - Click "Apply"

## Step 4: Add Models to Scene

1. **Create Parent GameObject**
   - In the Hierarchy panel (left), right-click
   - Select "Create Empty"
   - Rename it to "AssemblySteps"
   - Reset its position to (0, 0, 0) in Inspector

2. **Add Each Step Model**
   - From the Project panel, drag `sg50_step1` into the Hierarchy
   - Make it a child of "AssemblySteps" (drag onto AssemblySteps)
   - Repeat for all 7 steps (step1 through step7)
   - Position each at (0, 0, 0) so they overlap (we'll toggle visibility)

3. **Adjust Model Scale/Position if Needed**
   - Click "AssemblySteps" in Hierarchy
   - In Inspector, adjust Transform:
     - Position: (0, 0, 0)
     - Rotation: (0, 0, 0)
     - Scale: Start with (1, 1, 1), adjust if models are too small/large

## Step 5: Add Navigation Scripts

1. **Copy C# Scripts to Unity**
   - In Unity Project panel, right-click in Assets
   - Create → Folder, name it "Scripts"
   - Copy these files from `unity_scripts/` to Unity's `Assets/Scripts/`:
     - `AssemblyStepNavigator.cs`
     - `CameraController.cs`

2. **Add Step Navigator Script**
   - In Hierarchy, right-click → Create Empty
   - Rename it to "StepNavigator"
   - In Inspector, click "Add Component"
   - Search for "AssemblyStepNavigator" and add it
   - In the inspector, expand "Assembly Steps"
   - Set Size to 7
   - Drag each step model (step1-step7) from Hierarchy into the array slots

3. **Add Camera Controller**
   - In Hierarchy, click "Main Camera"
   - In Inspector, click "Add Component"
   - Search for "CameraController" and add it
   - Drag "AssemblySteps" from Hierarchy into the "Target" field

4. **Position Camera**
   - Select Main Camera in Hierarchy
   - Set Transform Position to (500, 300, -500) as a starting point
   - Rotation: (20, -45, 0)

## Step 6: Configure Lighting and Materials

1. **Add Lighting**
   - GameObject → Light → Directional Light (if not already present)
   - Position it above the models
   - Rotation: (50, -30, 0)

2. **Fix Pink Materials (if models appear pink)**
   - In Project panel, find your .obj files
   - Each will have a .mat (material) file next to it
   - Click on each material
   - In Inspector, change Shader to "Standard"
   - Adjust Albedo color to match LEGO colors

3. **Add a Background**
   - Window → Rendering → Lighting
   - Environment tab → Skybox Material
   - Choose a skybox or set background color

## Step 7: Test the Visualizer

1. **Enter Play Mode**
   - Click the ▶ Play button at top of Unity window
   - You should see Step 1 of your assembly

2. **Test Controls**
   - **Right Arrow / D**: Next step
   - **Left Arrow / A**: Previous step
   - **1-7 Keys**: Jump to specific step
   - **Space**: Toggle visibility
   - **Right Mouse Button**: Orbit camera
   - **Middle Mouse Button**: Pan camera
   - **Scroll Wheel**: Zoom in/out
   - **R**: Reset camera view
   - **F**: Focus on model

3. **Exit Play Mode**
   - Click the ▶ button again to stop

## Step 8: Optional Enhancements

### Add Grid Floor
```
GameObject → 3D Object → Plane
Scale: (100, 1, 100)
Position: (0, -10, 0)
```

### Better Camera Start Position
- Adjust Camera Controller settings in Inspector:
  - Orbit Speed: 100
  - Zoom Speed: 10
  - Min Distance: 100
  - Max Distance: 2000

### Add Background Color
```
Main Camera → Inspector → Background
Change from Skybox to Solid Color
Pick a color (e.g., light gray)
```

## Troubleshooting

### Models Don't Appear
- Check if models are at position (0, 0, 0)
- Try adjusting AssemblySteps scale (make it larger, e.g., 10, 10, 10)
- Check camera is pointing at origin

### Models Are Pink
- Pink means missing shader/material
- Select each material in Project panel
- Change Shader to "Standard"

### Controls Don't Work
- Make sure you're in Play mode (▶ button pressed)
- Check Console panel (Window → General → Console) for errors
- Verify scripts are attached correctly in Inspector

### Camera Is Too Close/Far
- Select Main Camera
- In Camera Controller component:
  - Increase Min/Max Distance values
  - Press R in Play mode to reset view

## Next Steps

Once you have the basic visualizer working, you can:

1. **Export Standalone Build**
   - File → Build Settings
   - Choose your platform (Mac, Windows, Linux)
   - Click "Build" to create an executable

2. **Add More Features**
   - Annotations showing part names
   - Animation between steps
   - Comparison mode (show two steps side-by-side)
   - VR support for immersive viewing

3. **Integrate with Your Research**
   - Use Unity ML-Agents for RL training
   - Connect to your Python pipeline via sockets
   - Record assembly sequences for dataset creation

## Quick Reference: File Locations

- **Your .obj models**: `data/processed/123456/unity_models/`
- **Unity scripts**: `unity_scripts/`
- **Unity project** (after creation): `~/Desktop/LEGO_Assembly_Visualizer/`

## Support Resources

- Unity Manual: https://docs.unity3d.com/Manual/
- Unity Learn: https://learn.unity.com/
- Unity Forums: https://forum.unity.com/

---

**You're all set!** Your LEGO assembly visualizer should now be working in Unity3D.

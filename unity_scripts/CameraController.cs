using UnityEngine;

/// <summary>
/// Simple camera controller for viewing LEGO models.
/// Attach this to your Main Camera.
/// </summary>
public class CameraController : MonoBehaviour
{
    [Header("Target")]
    [Tooltip("Optional: GameObject to orbit around. Leave empty to orbit around world origin.")]
    public Transform target;

    [Header("Orbit Settings")]
    [SerializeField]
    private float orbitSpeed = 100f;

    [SerializeField]
    private float zoomSpeed = 10f;

    [SerializeField]
    private float minDistance = 100f;

    [SerializeField]
    private float maxDistance = 2000f;

    [Header("Pan Settings")]
    [SerializeField]
    private float panSpeed = 0.5f;

    private float currentDistance = 500f;
    private float rotationX = 0f;
    private float rotationY = 30f;
    private Vector3 panOffset = Vector3.zero;

    private void Start()
    {
        // If no target specified, create an empty GameObject at origin
        if (target == null)
        {
            GameObject targetObj = new GameObject("CameraTarget");
            targetObj.transform.position = Vector3.zero;
            target = targetObj.transform;
        }

        // Initialize camera position
        UpdateCameraPosition();
    }

    private void Update()
    {
        HandleInput();
        UpdateCameraPosition();
    }

    private void HandleInput()
    {
        // Orbit with right mouse button
        if (Input.GetMouseButton(1))
        {
            rotationX += Input.GetAxis("Mouse X") * orbitSpeed * Time.deltaTime;
            rotationY -= Input.GetAxis("Mouse Y") * orbitSpeed * Time.deltaTime;
            rotationY = Mathf.Clamp(rotationY, -89f, 89f);
        }

        // Pan with middle mouse button
        if (Input.GetMouseButton(2))
        {
            float panX = -Input.GetAxis("Mouse X") * panSpeed * currentDistance * Time.deltaTime;
            float panY = -Input.GetAxis("Mouse Y") * panSpeed * currentDistance * Time.deltaTime;

            panOffset += transform.right * panX;
            panOffset += transform.up * panY;
        }

        // Zoom with scroll wheel
        float scroll = Input.GetAxis("Mouse ScrollWheel");
        if (Mathf.Abs(scroll) > 0.01f)
        {
            currentDistance -= scroll * zoomSpeed * currentDistance;
            currentDistance = Mathf.Clamp(currentDistance, minDistance, maxDistance);
        }

        // Reset view with R key
        if (Input.GetKeyDown(KeyCode.R))
        {
            ResetCamera();
        }

        // Focus on target with F key
        if (Input.GetKeyDown(KeyCode.F) && target != null)
        {
            panOffset = Vector3.zero;
        }
    }

    private void UpdateCameraPosition()
    {
        if (target == null) return;

        // Calculate position based on rotation and distance
        Quaternion rotation = Quaternion.Euler(rotationY, rotationX, 0);
        Vector3 offset = rotation * Vector3.back * currentDistance;

        // Apply position and rotation
        transform.position = target.position + panOffset + offset;
        transform.LookAt(target.position + panOffset);
    }

    private void ResetCamera()
    {
        rotationX = 0f;
        rotationY = 30f;
        currentDistance = 500f;
        panOffset = Vector3.zero;
        Debug.Log("Camera view reset");
    }

    private void OnGUI()
    {
        GUIStyle style = new GUIStyle(GUI.skin.label);
        style.fontSize = 14;
        style.normal.textColor = Color.white;
        style.alignment = TextAnchor.UpperRight;

        string info = "Camera Controls:\n";
        info += "  Right Mouse: Orbit\n";
        info += "  Middle Mouse: Pan\n";
        info += "  Scroll: Zoom\n";
        info += "  R: Reset view\n";
        info += "  F: Focus on target";

        GUI.Label(new Rect(Screen.width - 210, 10, 200, 120), info, style);
    }
}

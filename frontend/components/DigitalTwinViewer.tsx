"use client";

import { Suspense, useRef, useState } from "react";
import { Canvas, useLoader } from "@react-three/fiber";
import { OrbitControls, Grid } from "@react-three/drei";
import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader.js";
import * as THREE from "three";
import type { Brick } from "@/lib/api";

// LEGO color ID to RGB mapping
const LEGO_COLOR_MAP: Record<number, string> = {
  0: "#05131D",   // Black
  1: "#0055BF",   // Blue
  2: "#237841",   // Green
  3: "#008F9B",   // Dark Turquoise
  4: "#C91A09",   // Red
  5: "#C870A0",   // Dark Pink
  6: "#583927",   // Brown
  7: "#9BA19D",   // Light Gray
  8: "#6D6E5C",   // Dark Gray
  9: "#B4D2E3",   // Light Blue
  10: "#4B9F4A",  // Bright Green
  11: "#55A5AF",  // Light Turquoise
  12: "#F2705E",  // Salmon
  13: "#FC97AC",  // Pink
  14: "#F2CD37",  // Yellow
  15: "#FFFFFF",  // White
  16: "#C2DAB8",  // Light Green
  17: "#FBE696",  // Light Yellow
  18: "#E4CD9E",  // Tan
  19: "#C9CAE2",  // Light Violet
  20: "#81007B",  // Purple
  21: "#2032B0",  // Dark Blue Violet
  22: "#FE8A18",  // Orange
  23: "#E4ADC8",  // Magenta
  24: "#BBE90B",  // Lime
  25: "#958A73",  // Dark Tan
};

function BrickMesh({ brick, apiBase }: { brick: Brick; apiBase: string }) {
  const meshFile = brick.geometry_reference.mesh_file;
  // Ensure we use the full backend URL, not a relative path
  const meshUrl = `${apiBase}/meshes/${meshFile}`;
  const color = LEGO_COLOR_MAP[brick.color_id] || "#888888";

  console.log("Loading mesh from:", meshUrl); // Debug log

  // Load OBJ file
  const obj = useLoader(OBJLoader, meshUrl);

  // Clone and prepare the object
  const clonedObj = obj.clone();

  // Apply material to all meshes in the object
  const material = new THREE.MeshStandardMaterial({
    color: color,
    side: THREE.DoubleSide, // Render both sides of faces
    flatShading: false,
    metalness: 0.1,
    roughness: 0.6,
  });

  clonedObj.traverse((child) => {
    if (child instanceof THREE.Mesh) {
      child.material = material;
      child.castShadow = true;
      child.receiveShadow = true;
    }
  });

  // Create transformation matrix from pose_4x4
  const matrix = new THREE.Matrix4();
  const pose = brick.pose_4x4;
  matrix.set(
    pose[0][0], pose[0][1], pose[0][2], pose[0][3],
    pose[1][0], pose[1][1], pose[1][2], pose[1][3],
    pose[2][0], pose[2][1], pose[2][2], pose[2][3],
    pose[3][0], pose[3][1], pose[3][2], pose[3][3]
  );

  // LDraw uses a different coordinate system, need to apply conversion
  // LDraw: Y-up, front-facing -Z
  // Three.js: Y-up, front-facing -Z (same, but scale differs)
  const ldrawToThree = new THREE.Matrix4();
  ldrawToThree.makeScale(0.01, -0.01, -0.01); // Scale down and flip Y/Z
  matrix.premultiply(ldrawToThree);

  // Apply the transformation
  clonedObj.applyMatrix4(matrix);

  return <primitive object={clonedObj} />;
}

function Scene({ bricks, apiBase }: { bricks: Brick[]; apiBase: string }) {
  return (
    <>
      {/* Lighting */}
      <ambientLight intensity={0.6} />
      <directionalLight position={[10, 10, 5]} intensity={0.8} />
      <directionalLight position={[-10, -10, -5]} intensity={0.3} />

      {/* Grid helper */}
      <Grid
        args={[100, 100]}
        cellSize={10}
        cellThickness={0.5}
        cellColor="#6e6e6e"
        sectionSize={50}
        sectionThickness={1}
        sectionColor="#9d4b4b"
        fadeDistance={400}
        fadeStrength={1}
        followCamera={false}
        infiniteGrid={true}
      />

      {/* Render all bricks */}
      <Suspense fallback={null}>
        {bricks.map((brick) => (
          <BrickMesh key={brick.brick_id} brick={brick} apiBase={apiBase} />
        ))}
      </Suspense>

      {/* Camera controls */}
      <OrbitControls
        makeDefault
        minPolarAngle={0}
        maxPolarAngle={Math.PI / 2}
        enableDamping
        dampingFactor={0.05}
      />
    </>
  );
}

export default function DigitalTwinViewer({
  bricks,
  height = 500,
}: {
  bricks: Brick[];
  height?: number;
}) {
  const [error, setError] = useState<string | null>(null);
  // Use the same API base as the rest of the app, defaulting to backend port
  const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  console.log("DigitalTwinViewer API Base:", apiBase); // Debug log
  console.log("Environment NEXT_PUBLIC_API_URL:", process.env.NEXT_PUBLIC_API_URL); // Debug log

  if (bricks.length === 0) {
    return (
      <div
        className="flex items-center justify-center bg-gray-900 rounded-lg border border-gray-700"
        style={{ height: `${height}px` }}
      >
        <p className="text-gray-500 text-sm">No bricks to display</p>
      </div>
    );
  }

  return (
    <div
      className="bg-gray-900 rounded-lg border border-gray-700 overflow-hidden"
      style={{ height: `${height}px` }}
    >
      {error ? (
        <div className="flex items-center justify-center h-full">
          <p className="text-red-400 text-sm">{error}</p>
        </div>
      ) : (
        <Canvas
          camera={{ position: [50, 50, 50], fov: 60 }}
          onCreated={({ gl }) => {
            gl.setClearColor("#0a0a0a");
          }}
        >
          <Scene bricks={bricks} apiBase={apiBase} />
        </Canvas>
      )}
      <div className="absolute bottom-2 right-2 text-[10px] text-gray-600 bg-gray-900/80 px-2 py-1 rounded">
        Click + drag to rotate • Scroll to zoom • Right-click + drag to pan
      </div>
    </div>
  );
}

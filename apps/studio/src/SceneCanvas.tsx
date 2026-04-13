import { useMemo, useRef } from "react";
import { Bloom, EffectComposer, Noise, Vignette, ChromaticAberration } from "@react-three/postprocessing";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Line, QuadraticBezierLine, RoundedBox, Text, Stars, Sparkles } from "@react-three/drei";
import { a, useSpring } from "@react-spring/three";

const AnimatedLine = a(Line);
const AnimatedBezier = a(QuadraticBezierLine);
import type { TraceBundle, TraceFrame } from "@neuroloom/core";
import type { SelectionState } from "./state";
import * as THREE from "three";

type SceneCanvasProps = {
  bundle: TraceBundle;
  frame: TraceFrame;
  selection: SelectionState;
  onSelect(selection: SelectionState): void;
};

type FocusState = "neutral" | "selected" | "related" | "muted";

export function SceneCanvas({ bundle, frame, selection, onSelect }: SceneCanvasProps) {
  const camera = bundle.manifest.camera_presets.find((entry) => entry.id === frame.camera_anchor) ?? bundle.manifest.camera_presets[0]!;

  return (
    <div className="scene-stage">
      <Canvas
        gl={{ antialias: true, preserveDrawingBuffer: true }}
        dpr={[1, 1.8]}
        onCreated={({ gl }) => {
          gl.setClearColor("#050710");
          gl.toneMappingExposure = 1.04;
        }}
      >
        <SceneRoot bundle={bundle} frame={frame} camera={camera} selection={selection} onSelect={onSelect} />
      </Canvas>
    </div>
  );
}

function SceneRoot({
  bundle,
  frame,
  camera,
  selection,
  onSelect,
}: {
  bundle: TraceBundle;
  frame: TraceFrame;
  camera: TraceBundle["manifest"]["camera_presets"][number];
  selection: SelectionState;
  onSelect(selection: SelectionState): void;
}) {
  const nodeMap = useMemo(() => new Map(bundle.graph.nodes.map((node) => [node.id, node])), [bundle.graph.nodes]);
  const nodeStateMap = useMemo(() => new Map(frame.node_states.map((state) => [state.nodeId, state])), [frame.node_states]);
  const edgeStateMap = useMemo(() => new Map(frame.edge_states.map((state) => [state.edgeId, state])), [frame.edge_states]);
  const renderPayloadId =
    bundle.manifest.payload_catalog.find((entry) => entry.kind === "render" && frame.payload_refs.includes(entry.id))?.id ?? null;
  const renderPayload = renderPayloadId ? safeParsePayload(bundle.payloads.get(renderPayloadId)) : null;
  const selectedNodeId = selection?.kind === "node" ? selection.id : null;
  const selectedEdgeId = selection?.kind === "edge" ? selection.id : null;
  const relatedNodeIds = new Set<string>();
  const relatedEdgeIds = new Set<string>();

  if (selectedNodeId) {
    relatedNodeIds.add(selectedNodeId);
    bundle.graph.edges.forEach((edge) => {
      if (edge.source === selectedNodeId || edge.target === selectedNodeId) {
        relatedEdgeIds.add(edge.id);
        relatedNodeIds.add(edge.source);
        relatedNodeIds.add(edge.target);
      }
    });
  }

  if (selectedEdgeId) {
    const activeEdge = bundle.graph.edges.find((edge) => edge.id === selectedEdgeId);
    if (activeEdge) {
      relatedEdgeIds.add(activeEdge.id);
      relatedNodeIds.add(activeEdge.source);
      relatedNodeIds.add(activeEdge.target);
    }
  }

  const focusPosition = selectedNodeId
    ? (nodeMap.get(selectedNodeId)?.position ?? null)
    : selectedEdgeId
      ? getEdgeFocusPosition(bundle, selectedEdgeId, nodeMap)
      : null;

  return (
    <>
      <CameraRig position={camera.position} target={camera.target} focusTarget={focusPosition} />
      <color attach="background" args={["#050710"]} />
      <fog attach="fog" args={["#050710", 10, 35]} />
      <ambientLight intensity={0.8} color="#acc5f6" />
      <directionalLight position={[10, 15, 12]} intensity={2.4} color="#d7f6ff" />
      <pointLight position={[-7, 6, 8]} intensity={2.0} color="#15f0ff" />
      <pointLight position={[10, -4, 8]} intensity={1.5} color="#ffb45b" />
      <Stars radius={40} depth={0} count={3000} factor={6} saturation={1} fade speed={1.5} />
      <Sparkles count={150} scale={25} size={3} speed={0.4} opacity={0.6} color="#15f0ff" />
      <StageBackdrop family={bundle.manifest.family} />
      {focusPosition ? <SelectionAura family={bundle.manifest.family} position={vectorToTuple(focusPosition)} /> : null}
      <FamilySignatureLayer bundle={bundle} frame={frame} payload={renderPayload} nodeStateMap={nodeStateMap} />
      {bundle.graph.edges.map((edge) => {
        const source = nodeMap.get(edge.source);
        const target = nodeMap.get(edge.target);
        const state = edgeStateMap.get(edge.id);
        if (!source || !target || !state) {
          return null;
        }
        return (
          <EdgeFlow
            key={edge.id}
            family={bundle.manifest.family}
            from={vectorToTuple(source.position)}
            to={vectorToTuple(target.position)}
            state={state}
            focus={getEdgeFocusState(edge.id, selection, relatedEdgeIds)}
          />
        );
      })}
      {bundle.graph.nodes.map((node) => (
        <NodeGlyph
          key={node.id}
          family={bundle.manifest.family}
          label={node.label}
          type={node.type}
          position={vectorToTuple(node.position)}
          state={nodeStateMap.get(node.id)}
          focus={getNodeFocusState(node.id, selection, relatedNodeIds)}
          onClick={() => onSelect({ id: node.id, kind: "node" })}
        />
      ))}
      {bundle.manifest.family === "transformer" ? (
        <AttentionRibbonLayer bundle={bundle} payload={renderPayload} selection={selection} />
      ) : null}
      <EffectComposer>
        <Bloom luminanceThreshold={0.05} intensity={1.8} mipmapBlur />
        <Noise opacity={0.03} />
        <ChromaticAberration offset={[0.001, 0.001] as any} opacity={0.5} />
        <Vignette offset={0.25} darkness={0.7} />
      </EffectComposer>
    </>
  );
}

function FamilySignatureLayer({
  bundle,
  frame,
  payload,
  nodeStateMap,
}: {
  bundle: TraceBundle;
  frame: TraceFrame;
  payload: unknown;
  nodeStateMap: Map<string, TraceFrame["node_states"][number]>;
}) {
  if (bundle.manifest.family === "mlp") {
    return <MlpSignatureLayer frame={frame} payload={payload} />;
  }

  if (bundle.manifest.family === "cnn") {
    return <CnnSignatureLayer payload={payload} />;
  }

  return <TransformerSignatureLayer bundle={bundle} payload={payload} nodeStateMap={nodeStateMap} />;
}

function CameraRig({
  position,
  target,
  focusTarget,
}: {
  position: { x: number; y: number; z: number };
  target: { x: number; y: number; z: number };
  focusTarget: { x: number; y: number; z: number } | null;
}) {
  const { camera } = useThree();
  const baseTarget = useMemo(() => new THREE.Vector3(target.x, target.y, target.z), [target]);
  const focusVector = useMemo(() => (focusTarget ? new THREE.Vector3(focusTarget.x, focusTarget.y, focusTarget.z) : null), [focusTarget]);
  const cameraVector = useMemo(() => new THREE.Vector3(position.x, position.y, position.z), [position]);
  const lookTargetRef = useMemo(() => new THREE.Vector3(), []);

  useFrame(() => {
    camera.position.lerp(cameraVector, 0.08);
    lookTargetRef.copy(baseTarget);
    if (focusVector) {
      lookTargetRef.lerp(focusVector, 0.22);
    }
    camera.lookAt(lookTargetRef.x, lookTargetRef.y, lookTargetRef.z);
  });

  return null;
}

function SelectionAura({ family, position }: { family: TraceBundle["manifest"]["family"]; position: [number, number, number] }) {
  const ringScale = family === "transformer" ? 1.6 : family === "cnn" ? 1.2 : 1;
  const groupRef = useRef<THREE.Group>(null);

  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.rotation.z = state.clock.elapsedTime * 0.5;
      groupRef.current.scale.setScalar(1 + Math.sin(state.clock.elapsedTime * 3) * 0.05);
    }
  });

  return (
    <group position={[position[0], position[1], position[2] - 0.48]}>
      <group ref={groupRef}>
        <mesh>
          <ringGeometry args={[0.72 * ringScale, 1.04 * ringScale, 72]} />
          <meshBasicMaterial color="#d8ff66" transparent opacity={0.25} />
        </mesh>
        <mesh position={[0, 0, -0.06]}>
          <planeGeometry args={[2.1 * ringScale, 2.1 * ringScale]} />
          <meshBasicMaterial color="#15f0ff" transparent opacity={0.08} />
        </mesh>
      </group>
    </group>
  );
}

function StageBackdrop({ family }: { family: TraceBundle["manifest"]["family"] }) {
  if (family === "mlp") {
    return (
      <group>
        {[-5, -2, 1.4, 4.8].map((x, index) => (
          <mesh key={x} position={[x, 0, -1.4]}>
            <planeGeometry args={[2.4, 7.4]} />
            <meshBasicMaterial color={index % 2 === 0 ? "#0d1830" : "#101726"} transparent opacity={0.3} />
          </mesh>
        ))}
        <mesh position={[1.3, -3.4, -0.8]} rotation={[-0.24, 0.1, 0]}>
          <planeGeometry args={[5.5, 2.1]} />
          <meshBasicMaterial color="#12203a" transparent opacity={0.34} />
        </mesh>
      </group>
    );
  }

  if (family === "cnn") {
    return (
      <group>
        {[-6, -3.8, 0.8, 5.6].map((x, index) => (
          <mesh key={x} position={[x, 0, -1.8]}>
            <planeGeometry args={[2.2, 6.2]} />
            <meshBasicMaterial color={index % 2 === 0 ? "#0f1626" : "#101a30"} transparent opacity={0.28} />
          </mesh>
        ))}
        {[-0.45, -0.15, 0.15, 0.45].map((offset) => (
          <mesh key={offset} position={[-1.8 + offset, 2.9 - offset * 2, -0.4]}>
            <planeGeometry args={[1.2, 1.2]} />
            <meshBasicMaterial color="#17365b" transparent opacity={0.16} />
          </mesh>
        ))}
      </group>
    );
  }

  return (
    <group>
      <mesh position={[-5.2, 0, -1.5]}>
        <planeGeometry args={[1.7, 7.2]} />
        <meshBasicMaterial color="#11192e" transparent opacity={0.3} />
      </mesh>
      <mesh position={[0.2, 0, -1.6]}>
        <planeGeometry args={[2.3, 5.2]} />
        <meshBasicMaterial color="#112339" transparent opacity={0.26} />
      </mesh>
      <mesh position={[8.8, -0.2, -1.2]}>
        <planeGeometry args={[2, 3.8]} />
        <meshBasicMaterial color="#18223a" transparent opacity={0.32} />
      </mesh>
    </group>
  );
}

function EdgeFlow({
  family,
  from,
  to,
  state,
  focus,
}: {
  family: TraceBundle["manifest"]["family"];
  from: [number, number, number];
  to: [number, number, number];
  state: TraceFrame["edge_states"][number];
  focus: FocusState;
}) {
  const baseColor = state.direction === "backward" ? "#ffb45b" : "#15f0ff";
  const targetColor =
    focus === "selected"
      ? blendColor(baseColor, "#d8ff66", 0.58)
      : focus === "related"
        ? blendColor(baseColor, "#d8ff66", 0.22)
        : baseColor;
  const widthBoost = focus === "selected" ? 1.25 : focus === "related" ? 0.45 : focus === "muted" ? -0.2 : 0;
  const opacityMultiplier = focus === "muted" ? 0.18 : focus === "related" ? 0.74 : focus === "selected" ? 1 : 1;
  const targetWidth = Math.max(0.8, 1.2 + state.emphasis * 1.3 + widthBoost);
  const targetOpacity = clamp((0.14 + state.intensity * 0.42) * opacityMultiplier, 0.04, 0.92);

  const { color, width, opacity } = useSpring({
    color: targetColor,
    width: targetWidth,
    opacity: targetOpacity,
    config: { mass: 1, tension: 120, friction: 14 },
  });

  if (family === "transformer" && Math.abs(from[1] - to[1]) > 1.2) {
    const mid = [(from[0] + to[0]) / 2, Math.max(from[1], to[1]) + 1.5, (from[2] + to[2]) / 2] as [number, number, number];
    return (
      <QuadraticBezierLine
        start={from}
        end={to}
        mid={mid}
        color={color as any}
        lineWidth={width as any}
        transparent
        opacity={opacity as any}
      />
    );
  }

  return <Line points={[from, to]} color={color as any} lineWidth={width as any} transparent opacity={opacity as any} />;
}

function NodeGlyph({
  family,
  label,
  type,
  position,
  state,
  focus,
  onClick,
}: {
  family: TraceBundle["manifest"]["family"];
  label: string;
  type: string;
  position: [number, number, number];
  state: TraceFrame["node_states"][number] | undefined;
  focus: FocusState;
  onClick(): void;
}) {
  const activation = state?.activation ?? 0;
  const emphasis = state?.emphasis ?? 0.3;
  const baseColor = activation >= 0 ? "#15f0ff" : "#ffb45b";
  const targetHighlightColor = focus === "selected" ? "#d8ff66" : focus === "related" ? blendColor(baseColor, "#d8ff66", 0.18) : baseColor;
  const focusScale = focus === "selected" ? 1.14 : focus === "related" ? 1.04 : focus === "muted" ? 0.9 : 1;
  const targetOpacity = focus === "muted" ? 0.28 : focus === "related" ? 0.82 : 0.92;
  const targetEmissiveIntensity =
    focus === "selected" ? 2.25 + emphasis * 1.7 : focus === "related" ? 1.55 + emphasis * 1.35 : 0.9 + emphasis * 1.1;
  const sizeScale = (0.95 + emphasis * 0.55) * focusScale;
  const targetScale =
    family === "mlp"
      ? ([0.66 * sizeScale, 0.66 * sizeScale, 0.66 * sizeScale] as [number, number, number])
      : family === "cnn"
        ? ([1.05 * sizeScale, 0.42 * sizeScale, 0.28 + emphasis * 0.42] as [number, number, number])
        : ([1.22 * sizeScale, type === "token" ? 0.36 : 0.48, 0.18 + emphasis * 0.26] as [number, number, number]);

  const { emissive, emissiveIntensity, opacity, groupScale } = useSpring({
    emissive: targetHighlightColor,
    emissiveIntensity: targetEmissiveIntensity,
    opacity: targetOpacity,
    groupScale: targetScale,
    config: { mass: 1, tension: 150, friction: 18 },
  });

  return (
    <a.group position={position} scale={groupScale}>
      {family === "mlp" ? (
        <mesh onClick={onClick}>
          <sphereGeometry args={[0.34, 32, 32]} />
          <a.meshPhysicalMaterial
            color="#081520"
            emissive={emissive as any}
            emissiveIntensity={emissiveIntensity}
            roughness={0.08}
            metalness={0.15}
            transmission={0.4}
            thickness={0.5}
            clearcoat={1.0}
            clearcoatRoughness={0.1}
            transparent
            opacity={opacity}
          />
        </mesh>
      ) : (
        <RoundedBox args={[1, 1, 1]} radius={0.15} smoothness={6} onClick={onClick}>
          <a.meshPhysicalMaterial
            color="#07101b"
            emissive={emissive as any}
            emissiveIntensity={emissiveIntensity}
            roughness={0.12}
            metalness={0.2}
            transmission={0.3}
            thickness={0.4}
            clearcoat={0.8}
            clearcoatRoughness={0.1}
            transparent
            opacity={opacity}
          />
        </RoundedBox>
      )}
      {focus === "selected" ? (
        <mesh position={[0, 0, -0.2]}>
          <ringGeometry args={[0.45, 0.52, 48]} />
          <meshBasicMaterial color="#d8ff66" transparent opacity={0.66} />
        </mesh>
      ) : null}
      <Text
        position={[0, family === "transformer" ? -0.62 : -0.74, 0]}
        fontSize={0.18}
        color={focus === "muted" ? "#7f8ca8" : focus === "selected" ? "#fcffe1" : "#eef2ff"}
      >
        {label}
      </Text>
    </a.group>
  );
}

function AttentionRibbonLayer({ bundle, payload, selection }: { bundle: TraceBundle; payload: unknown; selection: SelectionState }) {
  if (!payload || typeof payload !== "object" || !("matrix" in payload)) {
    return null;
  }

  const matrix = Array.isArray(payload.matrix) ? payload.matrix : null;
  if (!matrix) return null;

  const tokens = bundle.graph.nodes.filter((node) => node.type === "token").sort((left, right) => left.order - right.order);
  const selectedTokenId = selection?.kind === "node" ? selection.id : null;

  return (
    <group>
      {tokens.flatMap((sourceNode, sourceIndex) =>
        tokens.map((targetNode, targetIndex) => {
          const weight = matrix[sourceIndex]?.[targetIndex];
          if (typeof weight !== "number" || weight < 0.32) {
            return null;
          }
          const start = vectorToTuple(sourceNode.position);
          const end = vectorToTuple(targetNode.position);
          const arcHeight = 1.4 + Math.abs(sourceIndex - targetIndex) * 0.55;
          const focused = selectedTokenId ? sourceNode.id === selectedTokenId || targetNode.id === selectedTokenId : false;
          const dimmed = Boolean(selectedTokenId) && !focused;
          return (
            <QuadraticBezierLine
              key={`${sourceNode.id}-${targetNode.id}`}
              start={start}
              end={end}
              mid={[(start[0] + end[0]) / 2, arcHeight, 0]}
              color={focused ? "#d8ff66" : "#15f0ff"}
              lineWidth={0.6 + weight * 1.1 + (focused ? 0.28 : 0)}
              transparent
              opacity={clamp((0.08 + weight * 0.28) * (dimmed ? 0.14 : focused ? 1.2 : 1), 0.03, 0.54)}
            />
          );
        }),
      )}
    </group>
  );
}

function MlpSignatureLayer({ frame, payload }: { frame: TraceFrame; payload: unknown }) {
  const matrix = readMatrix(payload);
  const series = readSeries(payload);
  const focusStrength = clamp(0.35 + (frame.metric_refs.find((metric) => metric.id === "confidence")?.value ?? 0.4) * 0.6, 0.25, 1);

  return (
    <group>
      <group position={[1.3, -3.15, 0.45]} rotation={[-1.18, 0.08, 0]}>
        <MatrixPlane matrix={sliceMatrix(matrix, 8, 8)} cellSize={0.24} depth={0.04} />
      </group>
      <SeriesBars3D position={[1.15, -4.05, 1.2]} series={series} color="#15f0ff" />
      <mesh position={[4.82, 0, -0.3]} rotation={[0, 0, Math.PI / 2]}>
        <ringGeometry args={[0.56, 0.72 + focusStrength * 0.1, 56]} />
        <meshBasicMaterial color="#15f0ff" transparent opacity={0.16 + focusStrength * 0.2} />
      </mesh>
      <mesh position={[4.82, 0, -0.42]} rotation={[0, 0, Math.PI / 2]}>
        <ringGeometry args={[0.84, 0.9 + focusStrength * 0.08, 56]} />
        <meshBasicMaterial color="#d8ff66" transparent opacity={0.08 + focusStrength * 0.12} />
      </mesh>
    </group>
  );
}

function CnnSignatureLayer({ payload }: { payload: unknown }) {
  const matrix = readMatrix(payload);
  const series = readSeries(payload);

  return (
    <group>
      <FeatureMapStack position={[-1.8, 2.65, 0.25]} matrix={sliceMatrix(matrix, 6, 6)} tint="#15f0ff" />
      <FeatureMapStack position={[3.08, 2.55, 0.18]} matrix={sliceMatrix(reverseColumns(matrix), 6, 6)} tint="#d8ff66" />
      <SeriesBars3D position={[8.5, -1.8, 0.7]} series={series} color="#ffb45b" />
      <RoundedBox args={[1.24, 2.42, 0.16]} radius={0.12} smoothness={4} position={[8.36, 0.18, -0.44]}>
        <meshBasicMaterial color="#172136" transparent opacity={0.24} />
      </RoundedBox>
    </group>
  );
}

function TransformerSignatureLayer({
  bundle,
  payload,
  nodeStateMap,
}: {
  bundle: TraceBundle;
  payload: unknown;
  nodeStateMap: Map<string, TraceFrame["node_states"][number]>;
}) {
  const series = readSeries(payload);
  const tokens = bundle.graph.nodes.filter((node) => node.type === "token").sort((left, right) => left.order - right.order);

  return (
    <group>
      <mesh position={[1.65, -1.42, -0.5]}>
        <planeGeometry args={[6.8, 0.24]} />
        <meshBasicMaterial color="#15f0ff" transparent opacity={0.1} />
      </mesh>
      <mesh position={[3.18, -1.42, -0.38]}>
        <planeGeometry args={[6.2, 0.14]} />
        <meshBasicMaterial color="#d8ff66" transparent opacity={0.08} />
      </mesh>
      <RoundedBox args={[3.2, 0.22, 0.1]} radius={0.08} smoothness={4} position={[1.8, -1.38, -0.1]}>
        <meshStandardMaterial color="#09111b" emissive="#15f0ff" emissiveIntensity={0.5} transparent opacity={0.72} />
      </RoundedBox>
      {tokens.map((token) => {
        const state = nodeStateMap.get(token.id);
        const emphasis = state?.emphasis ?? 0.5;
        return (
          <RoundedBox
            key={`token-plate-${token.id}`}
            args={[1.16, 0.18, 0.06]}
            radius={0.06}
            smoothness={4}
            position={[token.position.x, token.position.y - 0.42, -0.05]}
          >
            <meshStandardMaterial
              color="#08121d"
              emissive={"#15f0ff"}
              emissiveIntensity={0.25 + emphasis * 0.45}
              transparent
              opacity={0.68}
            />
          </RoundedBox>
        );
      })}
      <SeriesBars3D position={[9.3, -1.7, 0.55]} series={series} color="#15f0ff" />
      <RoundedBox args={[1.4, 2.7, 0.14]} radius={0.12} smoothness={4} position={[9.24, -0.1, -0.45]}>
        <meshBasicMaterial color="#162033" transparent opacity={0.24} />
      </RoundedBox>
    </group>
  );
}

function MatrixPlane({ matrix, cellSize, depth }: { matrix: number[][]; cellSize: number; depth: number }) {
  const rows = matrix.length;
  const columns = matrix[0]?.length ?? 0;
  const xOffset = ((columns - 1) * cellSize) / 2;
  const yOffset = ((rows - 1) * cellSize) / 2;

  return (
    <group>
      {matrix.flatMap((row, rowIndex) =>
        row.map((value, columnIndex) => {
          const positionX = columnIndex * cellSize - xOffset;
          const positionY = yOffset - rowIndex * cellSize;
          const magnitude = clamp(Math.abs(value), 0, 1);
          return (
            <mesh key={`${rowIndex}-${columnIndex}`} position={[positionX, positionY, magnitude * depth]}>
              <planeGeometry args={[cellSize * 0.82, cellSize * 0.82]} />
              <meshBasicMaterial
                color={blendColor("#121b2b", value >= 0 ? "#15f0ff" : "#ffb45b", 0.16 + magnitude * 0.84)}
                transparent
                opacity={0.24 + magnitude * 0.58}
              />
            </mesh>
          );
        }),
      )}
    </group>
  );
}

function FeatureMapStack({ position, matrix, tint }: { position: [number, number, number]; matrix: number[][]; tint: string }) {
  const layers = [0, 1, 2];
  return (
    <group position={position}>
      {layers.map((layer) => (
        <group key={layer} position={[layer * 0.16, -layer * 0.12, -layer * 0.15]}>
          <RoundedBox args={[1.6, 1.6, 0.04]} radius={0.08} smoothness={4}>
            <meshBasicMaterial color="#0d1422" transparent opacity={0.72 - layer * 0.12} />
          </RoundedBox>
          <group position={[0, 0, 0.05]}>
            <MatrixPlane matrix={matrix} cellSize={0.19} depth={0.02} />
          </group>
        </group>
      ))}
      <mesh position={[0, 0, -0.36]}>
        <planeGeometry args={[1.9, 1.9]} />
        <meshBasicMaterial color={tint} transparent opacity={0.08} />
      </mesh>
    </group>
  );
}

function SeriesBars3D({
  position,
  series,
  color,
}: {
  position: [number, number, number];
  series: Array<{ label: string; value: number }>;
  color: string;
}) {
  const items = series.slice(0, 3);
  return (
    <group position={position}>
      {items.map((item, index) => {
        const height = 0.25 + clamp(Math.abs(item.value), 0, 1.6) * 1.25;
        return (
          <group key={item.label} position={[index * 0.4 - ((items.length - 1) * 0.4) / 2, height / 2, 0]}>
            <RoundedBox args={[0.22, height, 0.22]} radius={0.06} smoothness={4}>
              <meshStandardMaterial color="#08111b" emissive={color} emissiveIntensity={0.7} transparent opacity={0.88} />
            </RoundedBox>
          </group>
        );
      })}
    </group>
  );
}

function safeParsePayload(raw: string | undefined) {
  if (!raw) return null;
  try {
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

function vectorToTuple(vector: { x: number; y: number; z: number }): [number, number, number] {
  return [vector.x, vector.y, vector.z];
}

function readMatrix(payload: unknown): number[][] {
  if (!payload || typeof payload !== "object" || !("matrix" in payload) || !Array.isArray(payload.matrix)) {
    return [
      [0.2, 0.35, 0.5, 0.12],
      [0.1, -0.12, 0.28, 0.42],
      [-0.34, 0.22, 0.45, 0.18],
      [0.15, 0.3, -0.18, 0.4],
    ];
  }

  return payload.matrix
    .filter((row): row is unknown[] => Array.isArray(row))
    .map((row) => row.map((value) => (typeof value === "number" ? value : 0)));
}

function readSeries(payload: unknown): Array<{ label: string; value: number }> {
  if (!payload || typeof payload !== "object" || !("series" in payload) || !Array.isArray(payload.series)) {
    return [
      { label: "signal", value: 0.42 },
      { label: "focus", value: 0.64 },
      { label: "drift", value: 0.28 },
    ];
  }

  return payload.series.flatMap((entry) => {
    if (!entry || typeof entry !== "object") return [];
    const label = "label" in entry && typeof entry.label === "string" ? entry.label : "metric";
    const value = "value" in entry && typeof entry.value === "number" ? entry.value : 0;
    return [{ label, value }];
  });
}

function sliceMatrix(matrix: number[][], maxRows: number, maxColumns: number) {
  return matrix.slice(0, maxRows).map((row) => row.slice(0, maxColumns));
}

function reverseColumns(matrix: number[][]) {
  return matrix.map((row) => [...row].reverse());
}

function blendColor(base: string, accent: string, amount: number) {
  return new THREE.Color(base).lerp(new THREE.Color(accent), clamp(amount, 0, 1)).getStyle();
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function getNodeFocusState(nodeId: string, selection: SelectionState, relatedNodeIds: Set<string>): FocusState {
  if (!selection) return "neutral";
  if (selection.kind === "node" && selection.id === nodeId) return "selected";
  return relatedNodeIds.has(nodeId) ? "related" : "muted";
}

function getEdgeFocusState(edgeId: string, selection: SelectionState, relatedEdgeIds: Set<string>): FocusState {
  if (!selection) return "neutral";
  if (selection.kind === "edge" && selection.id === edgeId) return "selected";
  return relatedEdgeIds.has(edgeId) ? "related" : "muted";
}

function getEdgeFocusPosition(bundle: TraceBundle, edgeId: string, nodeMap: Map<string, TraceBundle["graph"]["nodes"][number]>) {
  const edge = bundle.graph.edges.find((entry) => entry.id === edgeId);
  if (!edge) return null;
  const source = nodeMap.get(edge.source);
  const target = nodeMap.get(edge.target);
  if (!source || !target) return null;
  return {
    x: (source.position.x + target.position.x) / 2,
    y: (source.position.y + target.position.y) / 2,
    z: (source.position.z + target.position.z) / 2,
  };
}

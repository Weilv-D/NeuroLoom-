import { QuadraticBezierLine, Sparkles, Stars, Text } from "@react-three/drei";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Bloom, ChromaticAberration, EffectComposer, Noise, Vignette } from "@react-three/postprocessing";
import { type CSSProperties, useMemo, useRef } from "react";

import type { TraceBundle, TraceFrame } from "@neuroloom/core";
import type { QwenFramePayload, QwenSampleUnit } from "@neuroloom/official-traces";
import * as THREE from "three";

import type { SelectionState } from "./types";

type SceneCanvasProps = {
  bundle: TraceBundle;
  frame: TraceFrame | null;
  payload: QwenFramePayload | null;
  selection: SelectionState;
  onSelect(selection: SelectionState): void;
  live: boolean;
};

type FocusState = "selected" | "related" | "muted" | "neutral";

// ---------- vertex / fragment shaders for neuron point cloud ----------

const neuronVertexShader = /* glsl */ `
  attribute float aSize;
  attribute float aActivation;
  attribute float aIsAttn;
  attribute float aSelected;

  varying float vActivation;
  varying float vIsAttn;
  varying float vSelected;

  void main() {
    vActivation = aActivation;
    vIsAttn = aIsAttn;
    vSelected = aSelected;

    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = aSize * (220.0 / -mvPosition.z);
    gl_Position = projectionMatrix * mvPosition;
  }
`;

const neuronFragmentShader = /* glsl */ `
  varying float vActivation;
  varying float vIsAttn;
  varying float vSelected;

  void main() {
    // Circular soft particle
    vec2 center = gl_PointCoord - vec2(0.5);
    float dist = length(center);
    if (dist > 0.5) discard;

    float glow = 1.0 - smoothstep(0.0, 0.5, dist);

    float act = abs(vActivation);
    float intensity = act * glow;

    // Color mapping: star-galaxy palette
    vec3 darkBlue = vec3(0.04, 0.08, 0.13);      // inactive
    vec3 coldBlue = vec3(0.24, 0.91, 1.0);        // low activation
    vec3 white = vec3(0.93, 0.97, 1.0);           // mid activation
    vec3 warmGold = vec3(1.0, 0.98, 0.88);        // high activation
    vec3 amber = vec3(1.0, 0.72, 0.37);           // attention heads
    vec3 selectedGreen = vec3(0.84, 1.0, 0.39);   // selected

    vec3 color;
    if (vSelected > 0.5) {
      color = mix(selectedGreen, vec3(1.0), act * 0.4);
    } else if (vIsAttn > 0.5) {
      color = mix(darkBlue, amber, act);
    } else if (act < 0.15) {
      color = mix(darkBlue, coldBlue * 0.3, act / 0.15);
    } else if (act < 0.5) {
      color = mix(coldBlue, white, (act - 0.15) / 0.35);
    } else {
      color = mix(white, warmGold, (act - 0.5) / 0.5);
    }

    float alpha = mix(0.04, 0.92, act) * glow;
    if (vSelected > 0.5) alpha = max(alpha, 0.7);

    gl_FragColor = vec4(color * (0.3 + intensity * 1.4), alpha);
  }
`;

export function SceneCanvas({ bundle, frame, payload, selection, onSelect, live }: SceneCanvasProps) {
  const cameraPreset =
    bundle.manifest.camera_presets.find((entry) => entry.id === frame?.camera_anchor) ?? bundle.manifest.camera_presets[0]!;

  return (
    <div className="scene-stage scene-shell">
      <Canvas
        camera={{
          position: [cameraPreset.position.x, cameraPreset.position.y, cameraPreset.position.z],
          fov: cameraPreset.fov,
          near: 0.1,
          far: 120,
        }}
        style={{ width: "100%", height: "100%" }}
        gl={{ antialias: true, preserveDrawingBuffer: true }}
        dpr={[1, 1.8]}
        onCreated={({ gl }) => {
          gl.setClearColor("#04070d");
          gl.toneMappingExposure = 1.08;
        }}
      >
        <SceneRoot bundle={bundle} frame={frame} payload={payload} selection={selection} onSelect={onSelect} live={live} cameraPreset={cameraPreset} />
      </Canvas>
      <SceneOverlay bundle={bundle} frame={frame} payload={payload} selection={selection} onSelect={onSelect} live={live} />
    </div>
  );
}

function SceneRoot({
  bundle,
  frame,
  payload,
  selection,
  onSelect,
  live,
  cameraPreset,
}: {
  bundle: TraceBundle;
  frame: TraceFrame | null;
  payload: QwenFramePayload | null;
  selection: SelectionState;
  onSelect(selection: SelectionState): void;
  live: boolean;
  cameraPreset: TraceBundle["manifest"]["camera_presets"][number];
}) {
  const nodeMap = useMemo(() => new Map(bundle.graph.nodes.map((node) => [node.id, node])), [bundle.graph.nodes]);
  const nodeStateMap = useMemo(() => {
    if (!frame) {
      return new Map(
        bundle.graph.nodes.map((node) => [
          node.id,
          {
            nodeId: node.id,
            activation: node.type === "residual" ? 0.28 : node.type === "logits" ? 0.18 : 0.12,
            emphasis: node.type === "decode" ? 0.4 : 0.28,
          },
        ]),
      );
    }
    return new Map(frame.node_states.map((state) => [state.nodeId, state]));
  }, [bundle.graph.nodes, frame]);
  const edgeStateMap = useMemo(() => {
    if (!frame) {
      return new Map(bundle.graph.edges.map((edge) => [edge.id, { edgeId: edge.id, intensity: 0.16, direction: "forward", emphasis: 0.22 }]));
    }
    return new Map(frame.edge_states.map((state) => [state.edgeId, state]));
  }, [bundle.graph.edges, frame]);

  const focusedUnit = selection?.kind === "cluster" ? payload?.sampledUnits.find((unit) => unit.id === selection.id) ?? null : null;
  const focusedNodeId =
    selection?.kind === "node" ? selection.id : selection?.kind === "cluster" ? focusedUnit?.nodeId ?? null : selection?.kind === "token" ? "decode" : null;
  const connectedNodeIds = new Set<string>();
  const connectedEdgeIds = new Set<string>();

  if (focusedNodeId) {
    connectedNodeIds.add(focusedNodeId);
    bundle.graph.edges.forEach((edge) => {
      if (edge.source === focusedNodeId || edge.target === focusedNodeId) {
        connectedNodeIds.add(edge.source);
        connectedNodeIds.add(edge.target);
        connectedEdgeIds.add(edge.id);
      }
    });
  }

  const focusPosition = focusedUnit
    ? addOffset(nodeMap.get(focusedUnit.nodeId)?.position ?? { x: 0, y: 0, z: 0 }, clusterOffset(focusedUnit))
    : focusedNodeId
      ? nodeMap.get(focusedNodeId)?.position ?? null
      : null;

  const residualPoints = bundle.graph.nodes
    .filter((node) => node.metadata.lane === "residual")
    .map((node) => [node.position.x, node.position.y, node.position.z] as [number, number, number]);

  // Build simplified block-level flow edges
  const flowEdges = useMemo(() => {
    const result: { id: string; from: [number, number, number]; to: [number, number, number]; intensity: number }[] = [];
    const residualNodes = bundle.graph.nodes.filter((n) => n.metadata.lane === "residual");
    for (let i = 0; i < residualNodes.length - 1; i++) {
      const from = residualNodes[i]!;
      const to = residualNodes[i + 1]!;
      result.push({
        id: `flow-${i}`,
        from: [from.position.x, from.position.y, from.position.z],
        to: [to.position.x, to.position.y, to.position.z],
        intensity: 0.4,
      });
    }
    return result;
  }, [bundle.graph.nodes]);

  return (
    <>
      <CameraRig cameraPreset={cameraPreset} focusPosition={focusPosition} live={live} />
      <color attach="background" args={["#04070d"]} />
      <fog attach="fog" args={["#04070d", 14, 48]} />
      <ambientLight intensity={0.7} color="#cfe5ff" />
      <pointLight position={[-10, 6, 12]} intensity={2.4} color="#2fe5ff" />
      <pointLight position={[16, -6, 9]} intensity={1.8} color="#ffb85f" />
      <pointLight position={[0, 12, 14]} intensity={1.15} color="#d7ff63" />
      <Stars radius={50} depth={0} count={4200} factor={6} saturation={1} fade speed={1.2} />
      <Sparkles count={260} scale={[32, 18, 10]} size={3.8} speed={0.25} opacity={0.5} color="#1fe8ff" />
      <NebulaField />
      <ResidualRiver points={residualPoints} live={live} />

      <NeuronField
        graph={bundle.graph}
        frame={frame}
        selection={selection}
        onSelect={onSelect}
        live={live}
      />

      {/* Simplified block-level flow edges */}
      {flowEdges.map((fe) => (
        <FlowLine key={fe.id} from={fe.from} to={fe.to} live={live} />
      ))}

      {/* Retain attention/delta/ffn branch edges for structural context */}
      {bundle.graph.edges
        .filter((edge) => edge.type.includes("branch") || edge.type === "decode-flow" || edge.type === "token-flow")
        .map((edge) => {
          const source = nodeMap.get(edge.source);
          const target = nodeMap.get(edge.target);
          const state = edgeStateMap.get(edge.id);
          if (!source || !target || !state) return null;
          return (
            <EdgeStream
              key={edge.id}
              edgeId={edge.id}
              from={vectorToTuple(source.position)}
              to={vectorToTuple(target.position)}
              intensity={state.intensity}
              direction={state.direction}
              focus={focusForEdge(edge.id, selection, connectedEdgeIds)}
              live={live}
            />
          );
        })}

      {/* Structural anchor nodes (residual, prompt, decode, logits) */}
      {bundle.graph.nodes.map((graphNode) => {
        const nodeState = nodeStateMap.get(graphNode.id);
        const lane = String(graphNode.metadata.lane ?? graphNode.type);
        if (lane !== "residual" && lane !== "prompt" && lane !== "embedding" && graphNode.type !== "logits" && graphNode.type !== "decode") return null;
        return (
          <NodeAnchor
            key={graphNode.id}
            nodeId={graphNode.id}
            label={graphNode.label}
            type={graphNode.type}
            position={vectorToTuple(graphNode.position)}
            intensity={Math.abs(nodeState?.activation ?? 0.12)}
            emphasis={nodeState?.emphasis ?? 0.25}
            focus={focusForNode(graphNode.id, selection, connectedNodeIds)}
            onSelect={() => onSelect({ kind: "node", id: graphNode.id })}
            showLabel={selection?.kind === "node" && selection.id === graphNode.id}
          />
        );
      })}

      {payload ? <TokenRail payload={payload} selection={selection} onSelect={onSelect} /> : null}
      {payload ? <LogitWaterfall payload={payload} /> : null}
      {focusPosition ? <SelectionHalo position={vectorToTuple(focusPosition)} /> : null}
      <EffectComposer>
        <Bloom luminanceThreshold={0.02} intensity={2.1} mipmapBlur />
        <Noise opacity={0.025} />
        <ChromaticAberration offset={[0.0012, 0.0016] as [number, number]} />
        <Vignette offset={0.24} darkness={0.75} />
      </EffectComposer>
    </>
  );
}

// ---------- NeuronField: GPU point cloud for all neurons ----------

function NeuronField({
  graph,
  frame,
  selection,
  onSelect,
  live,
}: {
  graph: TraceBundle["graph"];
  frame: TraceFrame | null;
  selection: SelectionState;
  onSelect(selection: SelectionState): void;
  live: boolean;
}) {
  const pointsRef = useRef<THREE.Points>(null);

  // Build geometry once when graph changes
  const { geometry, neuronIds } = useMemo(() => {
    const neurons = graph.neurons ?? [];
    const positions = graph.neuronPositions ?? {};
    const count = neurons.length;

    const posArr = new Float32Array(count * 3);
    const ids: string[] = [];
    const attnArr = new Float32Array(count);

    for (let i = 0; i < count; i++) {
      const neuron = neurons[i]!;
      ids.push(neuron.id);
      const pos = positions[neuron.id] ?? [0, 0, 0];
      posArr[i * 3] = pos[0];
      posArr[i * 3 + 1] = pos[1];
      posArr[i * 3 + 2] = pos[2];
      attnArr[i] = neuron.lane === "attn_head" ? 1.0 : 0.0;
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(posArr, 3));
    geo.setAttribute("aSize", new THREE.BufferAttribute(new Float32Array(count), 1));
    geo.setAttribute("aActivation", new THREE.BufferAttribute(new Float32Array(count), 1));
    geo.setAttribute("aIsAttn", new THREE.BufferAttribute(attnArr, 1));
    geo.setAttribute("aSelected", new THREE.BufferAttribute(new Float32Array(count), 1));

    return { geometry: geo, neuronIds: ids };
  }, [graph]);

  // Build neuron state lookup
  const neuronStateMap = useMemo(() => {
    const map = new Map<string, number>();
    if (frame?.neuron_states) {
      for (const ns of frame.neuron_states) {
        map.set(ns.id, ns.activation);
      }
    }
    return map;
  }, [frame?.neuron_states]);

  // Update selection buffer only when selection changes
  const selectedId = selection?.kind === "neuron" ? selection.id : null;
  useMemo(() => {
    const count = neuronIds.length;
    const selAttr = geometry.getAttribute("aSelected") as THREE.BufferAttribute;
    for (let i = 0; i < count; i++) {
      selAttr.setX(i, neuronIds[i] === selectedId ? 1.0 : 0.0);
    }
    selAttr.needsUpdate = true;
  }, [geometry, neuronIds, selectedId]);

  // Per-frame update of size and activation buffers (dynamic)
  useFrame((state) => {
    const count = neuronIds.length;
    const sizeAttr = geometry.getAttribute("aSize") as THREE.BufferAttribute;
    const actAttr = geometry.getAttribute("aActivation") as THREE.BufferAttribute;

    const time = state.clock.elapsedTime;

    for (let i = 0; i < count; i++) {
      const id = neuronIds[i]!;
      let activation = neuronStateMap.get(id) ?? 0;

      // Pulse effect: add time-based shimmer for active neurons
      const act = Math.abs(activation);
      if (act > 0.1) {
        const shimmer = Math.sin(time * 3.0 + i * 0.007) * 0.06 * act;
        activation = activation + shimmer;
      }

      const absAct = Math.abs(activation);
      const baseSize = 0.006;
      const size = baseSize + absAct * absAct * 0.03;

      sizeAttr.setX(i, size);
      actAttr.setX(i, activation);
    }

    sizeAttr.needsUpdate = true;
    actAttr.needsUpdate = true;
  });

  const material = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: neuronVertexShader,
        fragmentShader: neuronFragmentShader,
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      }),
    [],
  );

  return (
    <points
      ref={pointsRef}
      geometry={geometry}
      material={material}
      onClick={(event) => {
        event.stopPropagation();

        // R3F raycasts Points objects and stores the hit index on event.index
        const idx = (event as unknown as { index?: number }).index;
        if (idx !== undefined && idx >= 0 && idx < neuronIds.length) {
          onSelect({ kind: "neuron", id: neuronIds[idx]! });
        }
      }}
      onPointerMissed={() => {
        // Don't clear selection on miss — keep the current selection
      }}
    />
  );
}

// ---------- FlowLine: simplified block-to-block flow ----------

function FlowLine({
  from,
  to,
  live,
}: {
  from: [number, number, number];
  to: [number, number, number];
  live: boolean;
}) {
  const pulseRef = useRef<THREE.Mesh>(null);
  const hash = useMemo(() => from[0] * 1000 + to[0] * 100, [from, to]);

  useFrame((state) => {
    if (!pulseRef.current) return;
    const speed = live ? 0.55 : 0.3;
    const t = (state.clock.elapsedTime * speed + hash * 0.00007) % 1;
    pulseRef.current.position.set(
      from[0] + (to[0] - from[0]) * t,
      from[1] + (to[1] - from[1]) * t,
      from[2] + (to[2] - from[2]) * t,
    );
    pulseRef.current.scale.setScalar(0.12);
  });

  return (
    <group>
      <QuadraticBezierLine
        start={from}
        mid={[(from[0] + to[0]) / 2, Math.max(from[1], to[1]) + 0.4, 0]}
        end={to}
        color="#2de2ff"
        lineWidth={1.8}
        transparent
        opacity={0.12}
      />
      <mesh ref={pulseRef}>
        <sphereGeometry args={[0.07, 8, 8]} />
        <meshBasicMaterial color="#2fe5ff" transparent opacity={0.4} />
      </mesh>
    </group>
  );
}

// ---------- CameraRig ----------

function CameraRig({
  cameraPreset,
  focusPosition,
  live,
}: {
  cameraPreset: TraceBundle["manifest"]["camera_presets"][number];
  focusPosition: { x: number; y: number; z: number } | null;
  live: boolean;
}) {
  const { camera } = useThree();
  const target = useMemo(() => new THREE.Vector3(cameraPreset.target.x, cameraPreset.target.y, cameraPreset.target.z), [cameraPreset.target]);
  const position = useMemo(
    () => new THREE.Vector3(cameraPreset.position.x, cameraPreset.position.y, cameraPreset.position.z),
    [cameraPreset.position],
  );
  const focusTarget = useMemo(() => (focusPosition ? new THREE.Vector3(focusPosition.x, focusPosition.y, focusPosition.z) : null), [focusPosition]);
  const lookAt = useMemo(() => new THREE.Vector3(), []);

  useFrame((state) => {
    const pulse = live ? Math.sin(state.clock.elapsedTime * 0.28) * 0.14 : 0;
    camera.position.lerp(new THREE.Vector3(position.x, position.y + pulse, position.z), 0.06);
    lookAt.copy(target);
    if (focusTarget) {
      lookAt.lerp(focusTarget, 0.16);
    }
    camera.lookAt(lookAt);
  });

  return null;
}

function NebulaField() {
  return (
    <group>
      {[
        { position: [-8, 4.5, -4], color: "#0a3a55", scale: [10, 5.5, 1.2], opacity: 0.28 },
        { position: [2, -3.5, -3], color: "#102743", scale: [18, 8, 1.4], opacity: 0.22 },
        { position: [12, 3.5, -5], color: "#3b2411", scale: [8, 4.5, 1], opacity: 0.16 },
      ].map((cloud) => (
        <mesh key={cloud.position.join(":")} position={cloud.position as [number, number, number]}>
          <planeGeometry args={[cloud.scale[0], cloud.scale[1]]} />
          <meshBasicMaterial color={cloud.color} transparent opacity={cloud.opacity} />
        </mesh>
      ))}
    </group>
  );
}

function ResidualRiver({ points, live }: { points: [number, number, number][]; live: boolean }) {
  const lineRef = useRef<THREE.Group>(null);

  useFrame((state) => {
    if (!lineRef.current) return;
    lineRef.current.position.y = live ? Math.sin(state.clock.elapsedTime * 0.7) * 0.08 : 0;
  });

  return (
    <group ref={lineRef}>
      <QuadraticBezierLine
        start={points[0] ?? [-10, 0, 0]}
        mid={[0, 0.55, 0]}
        end={points.at(-1) ?? [10, 0, 0]}
        color="#2de2ff"
        lineWidth={2.6}
        transparent
        opacity={0.18}
      />
      <QuadraticBezierLine
        start={points[0] ?? [-10, 0, 0]}
        mid={[0, -0.2, 0]}
        end={points.at(-1) ?? [10, 0, 0]}
        color="#d7ff63"
        lineWidth={1.2}
        transparent
        opacity={0.08}
      />
    </group>
  );
}

function EdgeStream({
  edgeId,
  from,
  to,
  intensity,
  direction,
  focus,
  live,
}: {
  edgeId: string;
  from: [number, number, number];
  to: [number, number, number];
  intensity: number;
  direction: string;
  focus: FocusState;
  live: boolean;
}) {
  const pulseRef = useRef<THREE.Mesh>(null);
  const arcMid = useMemo<[number, number, number]>(() => {
    const dx = to[0] - from[0];
    const dy = to[1] - from[1];
    const lift = Math.abs(dy) < 0.4 ? 1.2 : 0.55;
    return [from[0] + dx * 0.5, Math.max(from[1], to[1]) + lift, (from[2] + to[2]) / 2];
  }, [from, to]);
  const hash = useMemo(() => hashString(edgeId), [edgeId]);
  const opacity = focus === "muted" ? 0.05 : focus === "selected" || focus === "related" ? 0.42 : 0.18;
  const color = direction === "backward" ? "#ffb85f" : "#2fe5ff";

  useFrame((state) => {
    if (!pulseRef.current) return;
    const speed = live ? 0.55 : 0.3;
    const t = (state.clock.elapsedTime * speed + hash * 0.00007) % 1;
    const point = quadraticPoint(from, arcMid, to, t);
    pulseRef.current.position.set(point[0], point[1], point[2]);
    pulseRef.current.scale.setScalar(0.22 + intensity * 0.3);
  });

  return (
    <group>
      <QuadraticBezierLine
        start={from}
        mid={arcMid}
        end={to}
        color={color}
        lineWidth={focus === "selected" ? 2.3 : focus === "related" ? 1.5 : 1}
        transparent
        opacity={opacity + intensity * 0.16}
      />
      <mesh ref={pulseRef}>
        <sphereGeometry args={[0.09, 10, 10]} />
        <meshBasicMaterial color={color} transparent opacity={opacity + intensity * 0.22} />
      </mesh>
    </group>
  );
}

// ---------- NodeAnchor: simplified structural anchor (replaces NodeCluster for key nodes) ----------

function NodeAnchor({
  nodeId,
  label,
  type,
  position,
  intensity,
  emphasis,
  focus,
  onSelect,
  showLabel,
}: {
  nodeId: string;
  label: string;
  type: string;
  position: [number, number, number];
  intensity: number;
  emphasis: number;
  focus: FocusState;
  onSelect(): void;
  showLabel: boolean;
}) {
  const groupRef = useRef<THREE.Group>(null);
  const focusScale = focus === "selected" ? 1.18 : focus === "related" ? 1.08 : 1;

  useFrame((state) => {
    if (!groupRef.current) return;
    const wobble = 1 + Math.sin(state.clock.elapsedTime * 0.8 + intensity * 4.2) * 0.025;
    groupRef.current.scale.setScalar(focusScale * wobble);
  });

  return (
    <group ref={groupRef} position={position} onClick={(event) => {
      event.stopPropagation();
      onSelect();
    }}>
      <mesh>
        <sphereGeometry args={[0.09 + intensity * 0.15, 16, 16]} />
        <meshStandardMaterial
          color={type === "logits" || type === "decode" ? "#d7ff63" : "#2fe5ff"}
          emissive={type === "logits" || type === "decode" ? "#d7ff63" : "#2fe5ff"}
          emissiveIntensity={0.9 + emphasis * 1.6}
          transparent
          opacity={0.7}
        />
      </mesh>
      {showLabel || type === "decode" || type === "logits" ? (
        <Text
          position={[0, 0.48, 0]}
          fontSize={0.22}
          color="#f5f8ff"
          anchorX="center"
          anchorY="middle"
          outlineWidth={0.02}
          outlineColor="#04070d"
        >
          {label}
        </Text>
      ) : null}
    </group>
  );
}

function TokenRail({
  payload,
  selection,
  onSelect,
}: {
  payload: QwenFramePayload;
  selection: SelectionState;
  onSelect(selection: SelectionState): void;
}) {
  const startX = -15.2;
  const y = 6.1;
  return (
    <group>
      {payload.tokenWindow.map((token, index) => {
        const absoluteIndex = payload.tokenIndex - payload.tokenWindow.length + index + 1;
        const selected = selection?.kind === "token" && selection.id === `token-${absoluteIndex}`;
        const x = startX + index * 0.58;
        return (
          <group
            key={`${absoluteIndex}:${token}`}
            position={[x, y + Math.sin(index * 0.55) * 0.14, -1.2]}
            onClick={(event) => {
              event.stopPropagation();
              onSelect({ kind: "token", id: `token-${absoluteIndex}` });
            }}
          >
            <mesh>
              <sphereGeometry args={[selected ? 0.15 : 0.11, 14, 14]} />
              <meshBasicMaterial color={selected ? "#d7ff63" : index === payload.tokenWindow.length - 1 ? "#2fe5ff" : "#eef2ff"} transparent opacity={0.8} />
            </mesh>
            {selected || index === payload.tokenWindow.length - 1 ? (
              <Text
                position={[0, 0.34, 0]}
                fontSize={0.18}
                color="#eef2ff"
                anchorX="center"
                anchorY="middle"
                outlineWidth={0.02}
                outlineColor="#04070d"
              >
                {token.trim() || "space"}
              </Text>
            ) : null}
          </group>
        );
      })}
    </group>
  );
}

function LogitWaterfall({ payload }: { payload: QwenFramePayload }) {
  return (
    <group position={[15.5, -2.8, 0.4]}>
      {payload.topLogits.map((logit, index) => {
        const height = 0.4 + logit.score * 2.6;
        return (
          <group key={logit.token} position={[0, index * -0.8, 0]}>
            <mesh position={[0, height / 2, 0]}>
              <boxGeometry args={[0.24, height, 0.24]} />
              <meshStandardMaterial color={index === 0 ? "#d7ff63" : "#2fe5ff"} emissive={index === 0 ? "#d7ff63" : "#2fe5ff"} emissiveIntensity={0.7} />
            </mesh>
            <Text
              position={[0.64, 0.04, 0]}
              fontSize={0.18}
              color="#eef2ff"
              anchorX="left"
              anchorY="middle"
              outlineWidth={0.02}
              outlineColor="#04070d"
            >
              {`${logit.token.trim() || "space"} · ${logit.score.toFixed(2)}`}
            </Text>
          </group>
        );
      })}
    </group>
  );
}

function SelectionHalo({ position }: { position: [number, number, number] }) {
  const groupRef = useRef<THREE.Group>(null);
  useFrame((state) => {
    if (!groupRef.current) return;
    groupRef.current.rotation.z = state.clock.elapsedTime * 0.4;
    groupRef.current.scale.setScalar(1 + Math.sin(state.clock.elapsedTime * 3) * 0.06);
  });

  return (
    <group ref={groupRef} position={position}>
      <mesh>
        <ringGeometry args={[0.4, 0.54, 48]} />
        <meshBasicMaterial color="#d7ff63" transparent opacity={0.28} />
      </mesh>
    </group>
  );
}

// ---------- SceneOverlay: 2D HTML overlay ----------

function SceneOverlay({
  bundle,
  frame,
  payload,
  selection,
  onSelect,
  live,
}: {
  bundle: TraceBundle;
  frame: TraceFrame | null;
  payload: QwenFramePayload | null;
  selection: SelectionState;
  onSelect(selection: SelectionState): void;
  live: boolean;
}) {
  const nodeMap = useMemo(() => new Map(bundle.graph.nodes.map((node) => [node.id, node])), [bundle.graph.nodes]);
  const nodeStateMap = useMemo(
    () =>
      new Map(
        (frame?.node_states ?? bundle.graph.nodes.map((node) => ({ nodeId: node.id, activation: 0.16, emphasis: 0.2 }))).map((state) => [
          state.nodeId,
          state,
        ]),
      ),
    [bundle.graph.nodes, frame?.node_states],
  );
  const focusedUnit = selection?.kind === "cluster" ? payload?.sampledUnits.find((unit) => unit.id === selection.id) ?? null : null;
  const focusedNodeId =
    selection?.kind === "node" ? selection.id : selection?.kind === "cluster" ? focusedUnit?.nodeId ?? null : selection?.kind === "token" ? "decode" : null;
  const relatedNodeIds = new Set<string>();
  if (focusedNodeId) {
    relatedNodeIds.add(focusedNodeId);
    bundle.graph.edges.forEach((edge) => {
      if (edge.source === focusedNodeId || edge.target === focusedNodeId) {
        relatedNodeIds.add(edge.source);
        relatedNodeIds.add(edge.target);
      }
    });
  }

  const overlayEdges = bundle.graph.edges
    .filter((edge) => edge.type.includes("return") || edge.type === "decode-flow" || edge.type === "residual-flow")
    .slice(0, 120);

  return (
    <div className={`stage-overlay-2d ${live ? "is-live" : ""}`}>
      <svg className="stage-svg" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
        <defs>
          <linearGradient id="river" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#2fe5ff" stopOpacity="0.15" />
            <stop offset="55%" stopColor="#9ff4ff" stopOpacity="0.4" />
            <stop offset="100%" stopColor="#d7ff63" stopOpacity="0.18" />
          </linearGradient>
          <linearGradient id="flow" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#2fe5ff" stopOpacity="0.06" />
            <stop offset="100%" stopColor="#ffb85f" stopOpacity="0.22" />
          </linearGradient>
        </defs>
        <path d={residualPath(bundle)} className="stage-river" />
        {overlayEdges.map((edge) => {
          const source = nodeMap.get(edge.source);
          const target = nodeMap.get(edge.target);
          if (!source || !target) return null;
          return (
            <path
              key={edge.id}
              d={edgePath(source.position, target.position)}
              className={selection && (edge.source === focusedNodeId || edge.target === focusedNodeId) ? "stage-flow is-focused" : "stage-flow"}
            />
          );
        })}
      </svg>

      <div className="stage-stars">
        {bundle.graph.nodes.map((node) => {
          const state = nodeStateMap.get(node.id);
          const focus = focusForNode(node.id, selection, relatedNodeIds);
          const projected = project(node.position);
          return (
            <button
              key={node.id}
              type="button"
              className={`star-node lane-${String(node.metadata.lane ?? node.type)} focus-${focus}`}
              style={
                {
                  left: `${projected.left}%`,
                  top: `${projected.top}%`,
                  "--node-size": `${34 + Math.abs(state?.activation ?? 0.12) * 28}px`,
                  "--node-opacity": `${focus === "muted" ? 0.16 : 0.48 + Math.abs(state?.activation ?? 0.12) * 0.4}`,
                } as CSSProperties
              }
              onClick={() => onSelect({ kind: "node", id: node.id })}
            >
              <span className="star-core" />
              {Array.from({ length: 10 }, (_, index) => (
                <span
                  key={index}
                  className="star-grain"
                  style={grainStyle(node.id, index)}
                />
              ))}
              {selection?.kind === "node" && selection.id === node.id ? <span className="star-label">{node.label}</span> : null}
            </button>
          );
        })}

        {payload?.tokenWindow.map((token, index) => {
          const absoluteIndex = payload.tokenIndex - payload.tokenWindow.length + index + 1;
          const selected = selection?.kind === "token" && selection.id === `token-${absoluteIndex}`;
          return (
            <button
              key={`${absoluteIndex}:${token}`}
              type="button"
              className={selected ? "token-node is-selected" : "token-node"}
              style={{ left: `${12 + index * 2.8}%`, top: "10%" }}
              onClick={() => onSelect({ kind: "token", id: `token-${absoluteIndex}` })}
            >
              <span />
              {selected || index === payload.tokenWindow.length - 1 ? <em>{token.trim() || "space"}</em> : null}
            </button>
          );
        })}
      </div>

      {!payload ? <div className="stage-overlay">Waiting for token flow…</div> : null}
    </div>
  );
}

// ---------- Utility functions ----------

function focusForNode(nodeId: string, selection: SelectionState, relatedNodeIds: Set<string>): FocusState {
  if (!selection) return "neutral";
  if (selection.kind === "node" && selection.id === nodeId) return "selected";
  if (relatedNodeIds.has(nodeId)) return "related";
  return "muted";
}

function focusForEdge(edgeId: string, selection: SelectionState, relatedEdgeIds: Set<string>): FocusState {
  if (!selection) return "neutral";
  if (relatedEdgeIds.has(edgeId)) return "related";
  return "muted";
}

function clusterOffset(unit: QwenSampleUnit) {
  const laneOffsets: Record<QwenSampleUnit["lane"], [number, number, number]> = {
    residual: [0, 0.42, -0.18],
    attention: [-0.18, 0.56, 0.12],
    delta: [0.22, -0.52, -0.14],
    ffn: [0.16, -0.74, 0.18],
  };
  const base = laneOffsets[unit.lane];
  return {
    x: base[0] + (unit.cluster - 1) * 0.18,
    y: base[1] + Math.sin(unit.block * 0.6 + unit.cluster) * 0.08,
    z: base[2] + (unit.cluster - 1) * 0.06,
  };
}

function quadraticPoint(start: [number, number, number], mid: [number, number, number], end: [number, number, number], t: number) {
  const inv = 1 - t;
  return [
    inv * inv * start[0] + 2 * inv * t * mid[0] + t * t * end[0],
    inv * inv * start[1] + 2 * inv * t * mid[1] + t * t * end[1],
    inv * inv * start[2] + 2 * inv * t * mid[2] + t * t * end[2],
  ] as [number, number, number];
}

function hashString(value: string) {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index++) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function addOffset(position: { x: number; y: number; z: number }, offset: { x: number; y: number; z: number }) {
  return {
    x: position.x + offset.x,
    y: position.y + offset.y,
    z: position.z + offset.z,
  };
}

function vectorToTuple(vector: { x: number; y: number; z: number }): [number, number, number] {
  return [vector.x, vector.y, vector.z];
}

function project(position: { x: number; y: number; z: number }) {
  return {
    left: 50 + position.x * 2.65,
    top: 50 - position.y * 5.6,
  };
}

function residualPath(bundle: TraceBundle) {
  const residualNodes = bundle.graph.nodes.filter((node) => node.metadata.lane === "residual");
  return residualNodes
    .map((node, index) => {
      const point = project(node.position);
      return `${index === 0 ? "M" : "L"} ${point.left} ${point.top}`;
    })
    .join(" ");
}

function edgePath(source: { x: number; y: number; z: number }, target: { x: number; y: number; z: number }) {
  const start = project(source);
  const end = project(target);
  const midX = (start.left + end.left) / 2;
  const lift = Math.abs(start.top - end.top) < 6 ? -5 : -2;
  const midY = Math.min(start.top, end.top) + lift;
  return `M ${start.left} ${start.top} Q ${midX} ${midY} ${end.left} ${end.top}`;
}

function grainStyle(seed: string, index: number) {
  const rng = seeded(seed, index + 1);
  const left = 18 + rng() * 64;
  const top = 18 + rng() * 64;
  const delay = rng() * 1.8;
  const scale = 0.45 + rng() * 1.2;
  return {
    left: `${left}%`,
    top: `${top}%`,
    animationDelay: `${delay}s`,
    transform: `translate(-50%, -50%) scale(${scale})`,
  } satisfies CSSProperties;
}

function seeded(seed: string, salt: number) {
  let value = hashString(`${seed}:${salt}`) || 1;
  return () => {
    value = (value * 1664525 + 1013904223) >>> 0;
    return value / 4294967295;
  };
}

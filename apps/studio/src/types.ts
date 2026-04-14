export type SelectionState =
  | {
      kind: "node" | "cluster" | "token" | "neuron";
      id: string;
    }
  | null;

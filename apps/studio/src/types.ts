export type SelectionState =
  | {
      kind: "node" | "cluster" | "token";
      id: string;
    }
  | null;

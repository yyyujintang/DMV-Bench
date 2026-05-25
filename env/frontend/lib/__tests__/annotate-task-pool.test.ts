import { describe, expect, it, vi, beforeEach } from "vitest";

const findManyMock = vi.fn();
vi.mock("@/lib/prisma", () => ({
  prisma: {
    annotationSubTaskAttempt: {
      findMany: (...args: unknown[]) => findManyMock(...args),
    },
  },
}));

const readdirMock = vi.fn();
vi.mock("node:fs/promises", () => ({
  readdir: (...args: unknown[]) => readdirMock(...args),
}));

import { listTaskIds, pickTaskForWorker } from "@/lib/annotate/task-pool";

beforeEach(() => {
  findManyMock.mockReset();
  readdirMock.mockReset();
});

describe("listTaskIds", () => {
  it("returns sorted task ids without the .json suffix", async () => {
    readdirMock.mockResolvedValue(["nc_b.json", "nc_a.json", "reason.json", "notes.txt"]);
    const out = await listTaskIds("NC");
    expect(out).toEqual(["nc_a", "nc_b", "reason"]);
  });

  it("returns [] when the directory does not exist", async () => {
    const err: NodeJS.ErrnoException = new Error("nope");
    err.code = "ENOENT";
    readdirMock.mockRejectedValue(err);
    expect(await listTaskIds("NC")).toEqual([]);
  });
});

describe("pickTaskForWorker", () => {
  it("picks an unattempted task first", async () => {
    readdirMock.mockResolvedValue(["nc_a.json", "nc_b.json"]);
    findManyMock.mockResolvedValue([{ taskId: "nc_a", correct: false }]);
    const id = await pickTaskForWorker("NC", "worker_1");
    expect(id).toBe("nc_b");
  });

  it("cycles to an attempted-but-not-correct task when pool is exhausted", async () => {
    readdirMock.mockResolvedValue(["nc_a.json", "nc_b.json"]);
    findManyMock.mockResolvedValue([
      { taskId: "nc_a", correct: true },
      { taskId: "nc_b", correct: false },
    ]);
    const id = await pickTaskForWorker("NC", "worker_2");
    expect(id).toBe("nc_b");
  });

  it("returns null when the worker has nailed every task", async () => {
    readdirMock.mockResolvedValue(["nc_a.json"]);
    findManyMock.mockResolvedValue([{ taskId: "nc_a", correct: true }]);
    const id = await pickTaskForWorker("NC", "worker_3");
    expect(id).toBeNull();
  });

  it("returns null when the pool is empty", async () => {
    const err: NodeJS.ErrnoException = new Error("nope");
    err.code = "ENOENT";
    readdirMock.mockRejectedValue(err);
    findManyMock.mockResolvedValue([]);
    const id = await pickTaskForWorker("NC", "worker_4");
    expect(id).toBeNull();
  });
});

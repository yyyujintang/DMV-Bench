-- CreateTable
CREATE TABLE "AnnotationSubTaskAttempt" (
    "id" TEXT NOT NULL,
    "workerId" TEXT NOT NULL,
    "subTask" TEXT NOT NULL,
    "taskId" TEXT NOT NULL,
    "shopSessionId" TEXT NOT NULL,
    "startedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "finishedAt" TIMESTAMP(3),
    "navigationLog" TEXT NOT NULL DEFAULT '[]',
    "finalUrl" TEXT,
    "targetUrl" TEXT NOT NULL,
    "acceptedAlternatives" TEXT NOT NULL,
    "correct" BOOLEAN NOT NULL DEFAULT false,
    "gaveUp" BOOLEAN NOT NULL DEFAULT false,
    "totalDurationMs" INTEGER,

    CONSTRAINT "AnnotationSubTaskAttempt_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "AnnotationSubTaskAttempt_shopSessionId_key" ON "AnnotationSubTaskAttempt"("shopSessionId");

-- CreateIndex
CREATE INDEX "AnnotationSubTaskAttempt_workerId_idx" ON "AnnotationSubTaskAttempt"("workerId");

-- CreateIndex
CREATE INDEX "AnnotationSubTaskAttempt_subTask_idx" ON "AnnotationSubTaskAttempt"("subTask");

-- CreateIndex
CREATE INDEX "AnnotationSubTaskAttempt_taskId_idx" ON "AnnotationSubTaskAttempt"("taskId");

-- AddForeignKey
ALTER TABLE "AnnotationSubTaskAttempt" ADD CONSTRAINT "AnnotationSubTaskAttempt_workerId_fkey" FOREIGN KEY ("workerId") REFERENCES "AnnotationWorker"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

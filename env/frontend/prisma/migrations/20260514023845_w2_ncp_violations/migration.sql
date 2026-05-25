-- CreateTable
CREATE TABLE "NcpViolation" (
    "id" TEXT NOT NULL,
    "sessionId" TEXT,
    "turnIndex" INTEGER,
    "rule" TEXT NOT NULL,
    "violationType" TEXT NOT NULL,
    "description" TEXT NOT NULL,
    "renderedUrl" TEXT NOT NULL,
    "severity" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "NcpViolation_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "NcpViolation_rule_idx" ON "NcpViolation"("rule");

-- CreateIndex
CREATE INDEX "NcpViolation_createdAt_idx" ON "NcpViolation"("createdAt");

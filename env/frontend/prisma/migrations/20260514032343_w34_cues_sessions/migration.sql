-- CreateTable
CREATE TABLE "PeripheralCue" (
    "id" TEXT NOT NULL,
    "cueType" TEXT NOT NULL,
    "cueKey" TEXT NOT NULL,
    "visualDescriptor" TEXT NOT NULL,
    "assetUrl" TEXT NOT NULL,
    "defaultPosition" TEXT NOT NULL,
    "salienceScore" DOUBLE PRECISION NOT NULL DEFAULT 0.3,

    CONSTRAINT "PeripheralCue_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "TaskCueInjection" (
    "id" TEXT NOT NULL,
    "taskId" TEXT NOT NULL,
    "turnIndex" INTEGER NOT NULL,
    "pageId" TEXT NOT NULL,
    "cueId" TEXT NOT NULL,
    "positionOverride" TEXT,

    CONSTRAINT "TaskCueInjection_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "ShopSession" (
    "id" TEXT NOT NULL,
    "taskId" TEXT NOT NULL,
    "currentTurn" INTEGER NOT NULL DEFAULT 0,
    "mode" TEXT NOT NULL DEFAULT 'encoding',
    "anchorVariantIds" TEXT NOT NULL DEFAULT '',
    "viewHistory" TEXT NOT NULL DEFAULT '[]',
    "rejections" TEXT NOT NULL DEFAULT '[]',
    "wishlist" TEXT NOT NULL DEFAULT '[]',
    "taskSpec" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "endedAt" TIMESTAMP(3),

    CONSTRAINT "ShopSession_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "EncodingEvent" (
    "id" TEXT NOT NULL,
    "sessionId" TEXT NOT NULL,
    "turnIndex" INTEGER NOT NULL,
    "variantId" TEXT,
    "payload" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "EncodingEvent_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "PeripheralCue_cueKey_key" ON "PeripheralCue"("cueKey");

-- CreateIndex
CREATE INDEX "TaskCueInjection_taskId_turnIndex_idx" ON "TaskCueInjection"("taskId", "turnIndex");

-- CreateIndex
CREATE UNIQUE INDEX "TaskCueInjection_taskId_turnIndex_pageId_cueId_key" ON "TaskCueInjection"("taskId", "turnIndex", "pageId", "cueId");

-- CreateIndex
CREATE INDEX "ShopSession_taskId_idx" ON "ShopSession"("taskId");

-- CreateIndex
CREATE INDEX "EncodingEvent_sessionId_turnIndex_idx" ON "EncodingEvent"("sessionId", "turnIndex");

-- AddForeignKey
ALTER TABLE "TaskCueInjection" ADD CONSTRAINT "TaskCueInjection_cueId_fkey" FOREIGN KEY ("cueId") REFERENCES "PeripheralCue"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "EncodingEvent" ADD CONSTRAINT "EncodingEvent_sessionId_fkey" FOREIGN KEY ("sessionId") REFERENCES "ShopSession"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- CreateTable
CREATE TABLE "Category" (
    "id" TEXT NOT NULL,
    "slug" TEXT NOT NULL,
    "display" TEXT NOT NULL,
    "description" TEXT NOT NULL,
    "sortOrder" INTEGER NOT NULL DEFAULT 0,

    CONSTRAINT "Category_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Product" (
    "id" TEXT NOT NULL,
    "categoryId" TEXT NOT NULL,
    "baseName" TEXT NOT NULL,
    "baseDesc" TEXT NOT NULL,
    "longDesc" TEXT NOT NULL,
    "material" TEXT NOT NULL,
    "price" DOUBLE PRECISION NOT NULL,

    CONSTRAINT "Product_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "ProductVariant" (
    "id" TEXT NOT NULL,
    "productId" TEXT NOT NULL,
    "grainTier" INTEGER NOT NULL,
    "internalVariantKey" TEXT NOT NULL,
    "urlHash" TEXT NOT NULL,
    "primaryImage" TEXT NOT NULL,
    "galleryImages" TEXT NOT NULL,
    "title" TEXT NOT NULL,
    "altText" TEXT NOT NULL,

    CONSTRAINT "ProductVariant_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Review" (
    "id" TEXT NOT NULL,
    "variantId" TEXT NOT NULL,
    "rating" INTEGER NOT NULL,
    "body" TEXT NOT NULL,
    "authorName" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "Review_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "AnnotationWorker" (
    "id" TEXT NOT NULL,
    "prolificId" TEXT NOT NULL,
    "studyId" TEXT NOT NULL,
    "sessionId" TEXT,
    "sessionStartedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "sessionEndedAt" TIMESTAMP(3),
    "consentGiven" BOOLEAN NOT NULL DEFAULT false,
    "consentGivenAt" TIMESTAMP(3),
    "practiceCompleted" BOOLEAN NOT NULL DEFAULT false,
    "completionCode" TEXT,
    "status" TEXT NOT NULL DEFAULT 'in_progress',
    "rejectionReason" TEXT,
    "totalDurationMs" INTEGER,

    CONSTRAINT "AnnotationWorker_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Annotation4afc" (
    "id" TEXT NOT NULL,
    "workerId" TEXT NOT NULL,
    "taskId" TEXT,
    "anchorVariantId" TEXT NOT NULL,
    "candidateVariantIds" TEXT NOT NULL,
    "selectedVariantId" TEXT,
    "correct" BOOLEAN NOT NULL,
    "grainTier" INTEGER NOT NULL,
    "isAttentionCheck" BOOLEAN NOT NULL DEFAULT false,
    "responseTimeMs" INTEGER NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "Annotation4afc_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "AnnotationMechanism" (
    "id" TEXT NOT NULL,
    "workerId" TEXT NOT NULL,
    "taskId" TEXT NOT NULL,
    "mechanism" TEXT NOT NULL,
    "anchorVariantId" TEXT NOT NULL,
    "candidateVariantIds" TEXT NOT NULL,
    "fillerVariantIds" TEXT,
    "fillerLength" INTEGER,
    "selectedVariantId" TEXT,
    "groundTruthVariantId" TEXT NOT NULL,
    "correct" BOOLEAN NOT NULL,
    "grainTier" INTEGER NOT NULL,
    "anchorViewedDurationMs" INTEGER,
    "fillerViewedDurationMs" INTEGER,
    "responseTimeMs" INTEGER NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "AnnotationMechanism_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "Category_slug_key" ON "Category"("slug");

-- CreateIndex
CREATE UNIQUE INDEX "ProductVariant_urlHash_key" ON "ProductVariant"("urlHash");

-- CreateIndex
CREATE INDEX "ProductVariant_productId_grainTier_idx" ON "ProductVariant"("productId", "grainTier");

-- CreateIndex
CREATE UNIQUE INDEX "AnnotationWorker_prolificId_key" ON "AnnotationWorker"("prolificId");

-- CreateIndex
CREATE INDEX "Annotation4afc_workerId_idx" ON "Annotation4afc"("workerId");

-- CreateIndex
CREATE INDEX "Annotation4afc_anchorVariantId_idx" ON "Annotation4afc"("anchorVariantId");

-- CreateIndex
CREATE INDEX "Annotation4afc_grainTier_idx" ON "Annotation4afc"("grainTier");

-- CreateIndex
CREATE INDEX "AnnotationMechanism_workerId_idx" ON "AnnotationMechanism"("workerId");

-- CreateIndex
CREATE INDEX "AnnotationMechanism_taskId_idx" ON "AnnotationMechanism"("taskId");

-- CreateIndex
CREATE INDEX "AnnotationMechanism_mechanism_idx" ON "AnnotationMechanism"("mechanism");

-- AddForeignKey
ALTER TABLE "Product" ADD CONSTRAINT "Product_categoryId_fkey" FOREIGN KEY ("categoryId") REFERENCES "Category"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ProductVariant" ADD CONSTRAINT "ProductVariant_productId_fkey" FOREIGN KEY ("productId") REFERENCES "Product"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Review" ADD CONSTRAINT "Review_variantId_fkey" FOREIGN KEY ("variantId") REFERENCES "ProductVariant"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Annotation4afc" ADD CONSTRAINT "Annotation4afc_workerId_fkey" FOREIGN KEY ("workerId") REFERENCES "AnnotationWorker"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "AnnotationMechanism" ADD CONSTRAINT "AnnotationMechanism_workerId_fkey" FOREIGN KEY ("workerId") REFERENCES "AnnotationWorker"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

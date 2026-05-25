-- CreateTable
CREATE TABLE "CategorySubPage" (
    "id" TEXT NOT NULL,
    "categoryId" TEXT NOT NULL,
    "slug" TEXT NOT NULL,
    "title" TEXT NOT NULL,
    "displayOrder" INTEGER NOT NULL DEFAULT 0,

    CONSTRAINT "CategorySubPage_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "VariantSubPage" (
    "variantId" TEXT NOT NULL,
    "subPageId" TEXT NOT NULL,
    "position" INTEGER NOT NULL DEFAULT 0,

    CONSTRAINT "VariantSubPage_pkey" PRIMARY KEY ("variantId","subPageId")
);

-- CreateIndex
CREATE INDEX "CategorySubPage_categoryId_displayOrder_idx" ON "CategorySubPage"("categoryId", "displayOrder");

-- CreateIndex
CREATE UNIQUE INDEX "CategorySubPage_categoryId_slug_key" ON "CategorySubPage"("categoryId", "slug");

-- CreateIndex
CREATE INDEX "VariantSubPage_subPageId_position_idx" ON "VariantSubPage"("subPageId", "position");

-- AddForeignKey
ALTER TABLE "CategorySubPage" ADD CONSTRAINT "CategorySubPage_categoryId_fkey" FOREIGN KEY ("categoryId") REFERENCES "Category"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "VariantSubPage" ADD CONSTRAINT "VariantSubPage_variantId_fkey" FOREIGN KEY ("variantId") REFERENCES "ProductVariant"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "VariantSubPage" ADD CONSTRAINT "VariantSubPage_subPageId_fkey" FOREIGN KEY ("subPageId") REFERENCES "CategorySubPage"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

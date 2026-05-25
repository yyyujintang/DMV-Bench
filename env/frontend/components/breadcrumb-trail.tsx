/**
 * Opinionated breadcrumb that respects NCP rendering mode.
 *
 * Each crumb may carry an optional `thumbnail` (URL to a small visual
 * preview). In `encoding` mode the thumbnail renders next to the label;
 * in `recall` mode it's stripped per MR1 (proposal_website.md §5.1 point
 * 4) — breadcrumbs cannot leak anchor visual signal at recall time.
 *
 * W1 crumbs (category landing + sub-page list) don't yet carry thumbnails,
 * so the recall-mode behaviour is observably identical to encoding. The
 * seam exists for W2/W3 where a product-detail breadcrumb might include
 * the anchor thumbnail.
 *
 * Built on the existing `ui/breadcrumb.tsx` primitives so visual style
 * stays consistent across the site.
 */

import Link from "next/link";
import Image from "next/image";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";
import type { Mode } from "@/lib/mode";

export type Crumb = {
  label: string;
  href?: string;             // last crumb is typically current page → no href
  thumbnail?: string;        // small image; stripped in recall mode
  thumbnailAlt?: string;
};

export function BreadcrumbTrail({
  crumbs,
  mode,
}: {
  crumbs: Crumb[];
  mode: Mode;
}) {
  return (
    <Breadcrumb data-testid="breadcrumb-trail" data-mode={mode}>
      <BreadcrumbList>
        {crumbs.map((c, i) => {
          const isLast = i === crumbs.length - 1;
          const showThumb = mode === "encoding" && Boolean(c.thumbnail);
          return (
            <span key={`${c.label}-${i}`} className="inline-flex items-center gap-1.5">
              <BreadcrumbItem>
                {showThumb && c.thumbnail && (
                  <span
                    data-testid="breadcrumb-thumbnail"
                    className="relative inline-block h-5 w-5 overflow-hidden rounded-sm bg-stone-100"
                  >
                    <Image
                      src={c.thumbnail}
                      alt={c.thumbnailAlt ?? ""}
                      fill
                      sizes="20px"
                      className="object-cover"
                    />
                  </span>
                )}
                {isLast || !c.href ? (
                  <BreadcrumbPage>{c.label}</BreadcrumbPage>
                ) : (
                  <BreadcrumbLink render={<Link href={c.href} />}>
                    {c.label}
                  </BreadcrumbLink>
                )}
              </BreadcrumbItem>
              {!isLast && <BreadcrumbSeparator />}
            </span>
          );
        })}
      </BreadcrumbList>
    </Breadcrumb>
  );
}

# Management Buttons Guide

## Overview

The search page now has comprehensive management buttons for both AI text descriptions and CLIP image embeddings. These make it easy to keep your database synchronized and up-to-date.

## Button Sections

### 1. AI Text Descriptions

Generates detailed text descriptions using Watsonx AI.

**Buttons:**

- **Process New Items** - Generate descriptions only for items without AI descriptions
- **Force Regenerate All** - Regenerate descriptions for ALL items (uses API credits)

**When to use:**

- After adding new images to `data/assets/`
- When you want to update descriptions with newer AI models

---

### 2. CLIP Image Embeddings

Manages visual embeddings for fast and accurate image search.

**Buttons:**

- **🔄 Sync New Images** - Scan assets folder and add any new images to database
- **Generate Embeddings** - Generate CLIP embeddings only for items without embeddings
- **Force Regenerate All** - Regenerate embeddings for ALL items

**When to use:**

#### Sync New Images

- **Always run this first** after adding images to `data/assets/`
- Scans the assets folder and adds any new images to the database
- Creates basic database entries (descriptions will say "pending")
- Quick operation (milliseconds)

#### Generate Embeddings

- After syncing new images
- Generates CLIP embeddings for items that don't have them yet
- Takes a few seconds per image (first time loads the 605MB model)
- Required for CLIP image search to work

#### Force Regenerate All

- When you want to update embeddings with a newer CLIP model
- If embeddings got corrupted
- Generally not needed unless there's a specific reason

## Typical Workflow

### Adding New Images

```
1. Add images to data/assets/ folder
2. Click "🔄 Sync New Images"
   → Adds items to database
3. Click "Generate Embeddings"
   → Creates CLIP embeddings
4. (Optional) Click "Process New Items" in AI Text Descriptions
   → Generates detailed AI descriptions
```

### Current Status

After running the workflow above:

- **11 items** in database
- **11 CLIP embeddings** (100% coverage)
- Image search is **fully functional**

## API Endpoints

For developers who want to automate:

```bash
# Sync new images
curl -X POST http://localhost:8000/api/sync-assets

# Generate embeddings (new items only)
curl -X POST http://localhost:8000/api/generate-embeddings \
  -F "force=false"

# Force regenerate all embeddings
curl -X POST http://localhost:8000/api/generate-embeddings \
  -F "force=true"

# Generate AI descriptions (new items only)
curl -X POST http://localhost:8000/api/reprocess-descriptions \
  -F "force=false"
```

## Technical Details

### What Sync Does

1. Scans `data/assets/` for image files (.jpg, .jpeg, .png, .gif, .webp)
2. Compares against existing items in database
3. Adds new entries with:
   - Auto-generated ID (asset_001, asset_002, etc.)
   - Placeholder description
   - Basic metadata (color: "Various", material: "Mixed", etc.)
   - **No CLIP embedding yet** (needs separate generation)

### What Generate Embeddings Does

1. Loads CLIP model (once, ~1-2 seconds)
2. For each item without embedding:
   - Opens the image
   - Encodes to 512-dimensional vector
   - Saves to database
3. Model stays loaded between items (efficient batch processing)

### Performance

- **Sync:** ~5-10ms per image
- **Embedding Generation:** ~50-200ms per image (after model load)
- **Force Regenerate:** Same as above, but processes all items

## Troubleshooting

### "No items found to process"

- All items already have embeddings/descriptions
- Use force regenerate if you want to update anyway

### Image search not finding new images

1. Check database has the item: Look at search results count
2. Run "Sync New Images" if item is missing
3. Run "Generate Embeddings" if item exists but no results

### Model loading takes too long

- First time loads 605MB model (1-2 minutes on slow connection)
- Subsequent runs are fast (model is cached)
- Consider moving to GPU for 10x speed boost

## See Also

- [CLIP Upgrade Documentation](./CLIP_UPGRADE.md)
- [Main README](../README.md)

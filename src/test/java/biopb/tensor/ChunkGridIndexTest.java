package biopb.tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.junit.Assert;
import org.junit.Test;

import net.imglib2.FinalInterval;

/**
 * Unit tests for {@link ChunkGridIndex}, the aligned-grid endpoint index that
 * replaced three near-identical copies across the two Flight clients
 * (biopb/biopb#277 item D). Extraction is what lets the grid-completeness /
 * alignment rules be tested directly rather than only through a live server.
 *
 * <p>Tests use a 4x4 image on a 2x2 cell grid (four cells) and tag each chunk
 * with an {@code Integer} so lookups can be asserted by identity.
 */
public class ChunkGridIndexTest {

    private static final long[] DIMS = {4, 4};
    private static final int[] CELL = {2, 2};

    private static ChunkBounds bounds(long sx, long sy, long ex, long ey) {
        return ChunkBounds.newBuilder()
                .addStart(sx).addStart(sy)
                .addStop(ex).addStop(ey)
                .build();
    }

    // The four aligned 2x2 chunks tiling the 4x4 image, indexed by tag 0..3.
    private static final ChunkBounds[] TILES = {
        bounds(0, 0, 2, 2), // tag 0
        bounds(2, 0, 4, 2), // tag 1
        bounds(0, 2, 2, 4), // tag 2
        bounds(2, 2, 4, 4), // tag 3
    };

    private static ChunkGridIndex<Integer> buildFrom(List<Integer> tags) {
        return ChunkGridIndex.build(tags, DIMS, CELL, t -> TILES[t], t -> t);
    }

    private static FinalInterval cellAt(long x, long y) {
        return new FinalInterval(new long[] {x, y}, new long[] {x + 1, y + 1});
    }

    @Test
    public void complete_aligned_grid_builds_and_maps_each_cell() {
        ChunkGridIndex<Integer> idx = buildFrom(Arrays.asList(0, 1, 2, 3));
        Assert.assertNotNull(idx);
        Assert.assertEquals(Integer.valueOf(0), idx.get(idx.indexFor(cellAt(0, 0))));
        Assert.assertEquals(Integer.valueOf(1), idx.get(idx.indexFor(cellAt(2, 0))));
        Assert.assertEquals(Integer.valueOf(2), idx.get(idx.indexFor(cellAt(0, 2))));
        Assert.assertEquals(Integer.valueOf(3), idx.get(idx.indexFor(cellAt(2, 2))));
    }

    @Test
    public void incomplete_grid_returns_null() {
        // Only three of the four cells present -> not a complete grid.
        Assert.assertNull(buildFrom(Arrays.asList(0, 1, 2)));
    }

    @Test
    public void empty_items_returns_null() {
        Assert.assertNull(buildFrom(new ArrayList<>()));
    }

    @Test
    public void zero_dimensional_returns_null() {
        Assert.assertNull(ChunkGridIndex.build(
                Arrays.asList(0), new long[] {}, new int[] {}, t -> TILES[0], t -> t));
    }

    @Test
    public void misaligned_chunk_start_returns_null() {
        // A chunk whose x-start (1) is not a multiple of the cell width (2).
        List<ChunkBounds> items = Arrays.asList(
                bounds(1, 0, 3, 2), TILES[1], TILES[2], TILES[3]);
        Assert.assertNull(ChunkGridIndex.build(items, DIMS, CELL, b -> b, b -> b));
    }

    @Test
    public void wrong_sized_chunk_returns_null() {
        // Cell (0,0) present but sized 2x4 instead of the nominal 2x2.
        List<ChunkBounds> items = Arrays.asList(
                bounds(0, 0, 2, 4), TILES[1], TILES[2], TILES[3]);
        Assert.assertNull(ChunkGridIndex.build(items, DIMS, CELL, b -> b, b -> b));
    }

    @Test
    public void duplicate_cell_returns_null() {
        // Two chunks land on the same grid cell; the fourth cell is never filled.
        Assert.assertNull(buildFrom(Arrays.asList(0, 0, 1, 2)));
    }

    @Test
    public void indexFor_throws_on_unaligned_cell_min() {
        ChunkGridIndex<Integer> idx = buildFrom(Arrays.asList(0, 1, 2, 3));
        Assert.assertNotNull(idx);
        try {
            idx.indexFor(cellAt(1, 0)); // min x=1 is not aligned to cell width 2
            Assert.fail("expected IllegalStateException for unaligned cell min");
        } catch (IllegalStateException expected) {
            // ok
        }
    }

    @Test
    public void is_generic_over_the_stored_value_type() {
        // The same grid logic works for any value type (here String), which is
        // the whole reason the three concrete index classes could collapse.
        ChunkGridIndex<String> idx = ChunkGridIndex.build(
                Arrays.asList(0, 1, 2, 3), DIMS, CELL, t -> TILES[t], t -> "cell-" + t);
        Assert.assertNotNull(idx);
        Assert.assertEquals("cell-3", idx.get(idx.indexFor(cellAt(2, 2))));
    }
}

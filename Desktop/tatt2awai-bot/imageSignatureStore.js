const sharp = require('sharp');
const { createClient } = require('@supabase/supabase-js');
const logger = require('./logger');
const crypto = require('crypto');
const pLimit = require('p-limit');

const supabase = createClient(
    process.env.SUPABASE_URL,
    process.env.SUPABASE_KEY
);

const SIGNATURE_SETTINGS = {
    HASH_SIZE: 16,
    COLOR_BINS: 64,
    EDGE_SIZE: 32,
    BATCH_SIZE: 100,
    CONCURRENT_LIMIT: 5
};

class SignatureStore {
    constructor() {
        this.memoryCache = new Map();
        this.initPromise = null;
    }

    async initialize() {
        if (this.initPromise) return this.initPromise;

        this.initPromise = (async () => {
            const { data: signatures } = await supabase
                .from('image_signatures')
                .select('*');

            if (signatures) {
                signatures.forEach(sig => {
                    this.memoryCache.set(sig.path, sig.signature);
                });
            }
        })();

        return this.initPromise;
    }

    async generateSignature(buffer) {
        const [
            perceptualHash,
            colorInfo,
            edgeHash,
            metadata
        ] = await Promise.all([
            this.generatePerceptualHash(buffer),
            this.extractColorInfo(buffer),
            this.generateEdgeHash(buffer),
            sharp(buffer).metadata()
        ]);

        return {
            perceptualHash,
            colorInfo,
            edgeHash,
            metadata: {
                width: metadata.width,
                height: metadata.height,
                aspectRatio: metadata.width / metadata.height
            },
            generatedAt: Date.now()
        };
    }

    async generatePerceptualHash(buffer) {
        const data = await sharp(buffer)
            .resize(SIGNATURE_SETTINGS.HASH_SIZE, SIGNATURE_SETTINGS.HASH_SIZE)
            .grayscale()
            .raw()
            .toBuffer();

        const pixels = new Uint8Array(data);
        const mean = pixels.reduce((sum, val) => sum + val, 0) / pixels.length;
        return pixels.map(p => p > mean ? '1' : '0').join('');
    }

    async extractColorInfo(buffer) {
        const data = await sharp(buffer)
            .resize(32, 32)
            .raw()
            .toBuffer();

        const histogram = new Array(SIGNATURE_SETTINGS.COLOR_BINS).fill(0);
        const avgColor = { r: 0, g: 0, b: 0 };

        for (let i = 0; i < data.length; i += 3) {
            const r = data[i], g = data[i + 1], b = data[i + 2];
            const bin = Math.floor((r + g + b) / (768 / SIGNATURE_SETTINGS.COLOR_BINS));
            histogram[bin]++;
            avgColor.r += r;
            avgColor.g += g;
            avgColor.b += b;
        }

        const pixelCount = data.length / 3;
        avgColor.r /= pixelCount;
        avgColor.g /= pixelCount;
        avgColor.b /= pixelCount;

        return { histogram, avgColor };
    }

    async generateEdgeHash(buffer) {
        const data = await sharp(buffer)
            .resize(SIGNATURE_SETTINGS.EDGE_SIZE, SIGNATURE_SETTINGS.EDGE_SIZE)
            .grayscale()
            .convolve({
                width: 3,
                height: 3,
                kernel: [-1, -1, -1, -1, 8, -1, -1, -1, -1]
            })
            .normalize()
            .raw()
            .toBuffer();

        return Buffer.from(data).toString('base64');
    }

    async precomputeSignatures(dropboxManager) {
        const allImages = await dropboxManager.fetchDropboxEntries('');
        const limit = pLimit(SIGNATURE_SETTINGS.CONCURRENT_LIMIT);
        let processed = 0;

        // Process in batches
        for (let i = 0; i < allImages.result.entries.length; i += SIGNATURE_SETTINGS.BATCH_SIZE) {
            const batch = allImages.result.entries.slice(i, i + SIGNATURE_SETTINGS.BATCH_SIZE);
            
            await Promise.all(batch.map(image => limit(async () => {
                try {
                    if (!this.memoryCache.has(image.path_lower)) {
                        const buffer = await dropboxManager.downloadFile(image.path_lower);
                        if (!buffer?.result?.fileBinary) return;

                        const imageBuffer = Buffer.isBuffer(buffer.result.fileBinary) ?
                            buffer.result.fileBinary :
                            Buffer.from(buffer.result.fileBinary);

                        const signature = await this.generateSignature(imageBuffer);
                        
                        await supabase
                            .from('image_signatures')
                            .upsert({
                                path: image.path_lower,
                                signature,
                                updated_at: new Date()
                            });

                        this.memoryCache.set(image.path_lower, signature);
                    }
                } catch (error) {
                    logger.warn(`Failed to generate signature for ${image.path_lower}:`, error);
                }
                processed++;
                if (processed % 100 === 0) {
                    logger.info(`Processed ${processed}/${allImages.result.entries.length} images`);
                }
            })));
        }
    }

    async compareSignatures(sig1, sig2) {
        const hashScore = this.compareHashes(sig1.perceptualHash, sig2.perceptualHash);
        const colorScore = this.compareColors(sig1.colorInfo, sig2.colorInfo);
        const edgeScore = this.compareEdgeHashes(sig1.edgeHash, sig2.edgeHash);
        const aspectScore = this.compareAspectRatios(
            sig1.metadata.aspectRatio,
            sig2.metadata.aspectRatio
        );

        return {
            total: (hashScore * 0.4 + colorScore * 0.3 + edgeScore * 0.2 + aspectScore * 0.1),
            components: {
                hash: hashScore,
                color: colorScore,
                edge: edgeScore,
                aspect: aspectScore
            }
        };
    }

    compareHashes(hash1, hash2) {
        let diff = 0;
        for (let i = 0; i < hash1.length; i++) {
            if (hash1[i] !== hash2[i]) diff++;
        }
        return 1 - (diff / hash1.length);
    }

    compareColors(color1, color2) {
        const histScore = this.compareHistograms(color1.histogram, color2.histogram);
        const avgScore = this.compareAverageColors(color1.avgColor, color2.avgColor);
        return (histScore + avgScore) / 2;
    }

    compareHistograms(hist1, hist2) {
        const sum1 = hist1.reduce((a, b) => a + b, 0);
        const sum2 = hist2.reduce((a, b) => a + b, 0);
        
        return hist1.reduce((score, count, i) => {
            const h1 = count / sum1;
            const h2 = hist2[i] / sum2;
            return score + Math.min(h1, h2);
        }, 0);
    }

    compareAverageColors(color1, color2) {
        const diff = Math.sqrt(
            Math.pow(color1.r - color2.r, 2) +
            Math.pow(color1.g - color2.g, 2) +
            Math.pow(color1.b - color2.b, 2)
        );
        return 1 - (diff / 441.67); // sqrt(255^2 * 3)
    }

    compareEdgeHashes(edge1, edge2) {
        const buf1 = Buffer.from(edge1, 'base64');
        const buf2 = Buffer.from(edge2, 'base64');
        
        let diff = 0;
        for (let i = 0; i < buf1.length; i++) {
            diff += Math.abs(buf1[i] - buf2[i]);
        }
        return 1 - (diff / (buf1.length * 255));
    }

    compareAspectRatios(ratio1, ratio2) {
        return 1 - Math.abs(ratio1 - ratio2) / Math.max(ratio1, ratio2);
    }

    get(path) {
        return this.memoryCache.get(path);
    }

    set(path, signature) {
        this.memoryCache.set(path, signature);
    }
}

module.exports = new SignatureStore();

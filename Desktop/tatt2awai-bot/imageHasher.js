const sharp = require('sharp');
const crypto = require('crypto');

class ImageHasher {
    static async generateHash(buffer, size = 16) {
        try {
            // Process image
            const processedBuffer = await sharp(buffer)
                .greyscale()  // Convert to grayscale
                .resize(size, size, {  // Resize to target dimensions
                    fit: 'fill'
                })
                .raw()  // Get raw pixel data
                .toBuffer();

            // Convert to binary string based on average
            const pixels = Array.from(processedBuffer);
            const average = pixels.reduce((sum, val) => sum + val, 0) / pixels.length;
            
            // Generate binary hash
            let hash = '';
            for (const pixel of pixels) {
                hash += pixel >= average ? '1' : '0';
            }

            return hash;
        } catch (error) {
            throw new Error(`Error generating image hash: ${error.message}`);
        }
    }

    static async generateMultiHash(buffer, sizes = [16, 32, 64]) {
        try {
            // Generate hashes at different sizes in parallel
            const hashes = await Promise.all(sizes.map(async size => ({
                size,
                hash: await this.generateHash(buffer, size)
            })));

            return hashes;
        } catch (error) {
            throw new Error(`Error generating multi-hash: ${error.message}`);
        }
    }

    static calculateSimilarity(hash1, hash2) {
        if (hash1.length !== hash2.length) {
            throw new Error('Hashes must be the same length');
        }

        let differences = 0;
        for (let i = 0; i < hash1.length; i++) {
            if (hash1[i] !== hash2[i]) {
                differences++;
            }
        }

        return 1 - (differences / hash1.length);
    }

    static async compareImages(buffer1, buffer2, size = 16) {
        const hash1 = await this.generateHash(buffer1, size);
        const hash2 = await this.generateHash(buffer2, size);
        return this.calculateSimilarity(hash1, hash2);
    }

    static async compareMultipleResolutions(buffer1, buffer2, sizes = [16, 32, 64]) {
        const hashes1 = await this.generateMultiHash(buffer1, sizes);
        const hashes2 = await this.generateMultiHash(buffer2, sizes);

        const similarities = hashes1.map((hash1, index) => {
            const hash2 = hashes2[index];
            return {
                size: hash1.size,
                similarity: this.calculateSimilarity(hash1.hash, hash2.hash)
            };
        });

        // Calculate weighted average (larger sizes have more weight)
        const totalWeight = sizes.reduce((sum, size) => sum + size, 0);
        const weightedSimilarity = similarities.reduce((sum, { size, similarity }) => 
            sum + (similarity * size / totalWeight), 0);

        return {
            similarities,
            averageSimilarity: weightedSimilarity
        };
    }
}

module.exports = ImageHasher;

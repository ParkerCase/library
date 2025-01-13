// bufferUtils.js
const sharp = require('sharp');
const logger = require('./logger');

class BufferUtils {
    static async validateAndConvert(buffer, options = {}) {
        try {
            if (!Buffer.isBuffer(buffer)) {
                if (ArrayBuffer.isView(buffer)) {
                    buffer = Buffer.from(buffer.buffer);
                } else {
                    throw new Error('Invalid input: Buffer expected');
                }
            }

            // For raw buffer handling
            if (options.raw) {
                return {
                    buffer,
                    metadata: {
                        width: options.width,
                        height: options.height,
                        channels: options.channels || 1,
                        format: 'raw'
                    }
                };
            }

            // Convert to standard format
            const processedBuffer = await this.processImageBuffer(buffer);
            return processedBuffer;
        } catch (error) {
            logger.error('Buffer validation failed:', {
                error: error.message,
                bufferType: typeof buffer,
                isBuffer: Buffer.isBuffer(buffer),
                size: buffer?.length || 0
            });
            throw error;
        }
    }

    static async processImageBuffer(buffer) {
        const sharpOptions = {
            failOnError: false,
            unlimited: true,
            sequentialRead: true,
            density: 300
        };

        const image = sharp(buffer, sharpOptions);
        const metadata = await image.metadata();

        const processedBuffer = await image
            .rotate()
            .removeAlpha()
            .resize(2048, 2048, {
                fit: 'inside',
                withoutEnlargement: true,
                kernel: 'lanczos3'
            })
            .jpeg({
                quality: 90,
                chromaSubsampling: '4:4:4',
                force: true,
                mozjpeg: true
            })
            .toBuffer();

        const processedMetadata = await sharp(processedBuffer, sharpOptions).metadata();

        return {
            buffer: processedBuffer,
            metadata: processedMetadata
        };
    }

    static async ensureValidRawBuffer(data, width, height, channels = 1) {
        return this.validateAndConvert(data, {
            raw: true,
            width,
            height,
            channels
        });
    }
}

module.exports = BufferUtils;

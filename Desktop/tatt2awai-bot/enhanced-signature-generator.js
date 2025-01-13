const sharp = require("sharp");
const _ = require("lodash");
const crypto = require("crypto");
const config = require("./config");
const monitor = require("./monitoring");
const cache = require("./cacheManager");
const { ValidationError, ProcessingError } = require("./errorHandler");
const ColorAnalysis = require("./colorAnalysis");
const EdgeAnalysis = require("./edgeAnalysis");
const TextureAnalysis = require("./textureAnalysis");
const logger = require("./logger");
const mathUtils = require("./mathUtils");

// Configure sharp globally for better performance
sharp.cache(false);
sharp.concurrency(1);

class BufferManager {
  constructor() {
    this.buffers = new Map();
  }

  async ensureBuffer(buffer) {
    if (!Buffer.isBuffer(buffer)) {
      throw new Error("Invalid input: Buffer expected");
    }

    // Create a hash of the buffer to use as a key
    const key = this.createBufferHash(buffer);

    if (!this.buffers.has(key)) {
      // Make a defensive copy
      this.buffers.set(key, Buffer.from(buffer));
    }

    return {
      buffer: this.buffers.get(key),
      key,
    };
  }

  getBuffer(key) {
    return this.buffers.get(key);
  }

  releaseBuffer(key) {
    if (this.buffers.has(key)) {
      this.buffers.delete(key);
    }
  }

  createBufferHash(buffer) {
    const crypto = require("crypto");
    return crypto.createHash("sha256").update(buffer).digest("hex");
  }

  clear() {
    this.buffers.clear();
  }
}

class EnhancedSignatureGenerator {
  constructor(options = {}) {
    this.bufferManager = new BufferManager();
    this.config = {
      COLOR: {
        DOMINANT_COLOR_COUNT: 8,
        COLOR_QUANT_LEVELS: {
          RGB: 64,
          LAB: 32,
          HSV: 48,
        },
        HISTOGRAM_SMOOTHING: 0.1,
      },
      EDGE: {
        CANNY_LOW: 0.1,
        CANNY_HIGH: 0.3,
        ORIENTATION_BINS: 36,
        MIN_EDGE_LENGTH: 5,
      },
      HASH: {
        DCT_SIZE: 32,
        HASH_SIZE: 256,
        ERROR_CORRECTION_LEVEL: "medium",
      },
      TEXTURE: {
        GABOR_ORIENTATIONS: [0, 30, 60, 90, 120, 150],
        GABOR_FREQUENCIES: [0.1, 0.2, 0.3, 0.4],
        LBP_RADIUS: 2,
        GLCM_DISTANCES: [1, 2, 4],
      },
      ...options,
    };

    // Configure sharp globally
    sharp.cache(false);
    sharp.concurrency(1);
    sharp.simd(true);

    this.initGaloisTables();

    // Configure buffer processing settings
    this.processingOptions = {
      failOnError: false,
      unlimited: true,
      sequentialRead: true,
      pages: 1,
      animated: false,
      limitInputPixels: 512 * 512 * 1024, // 512MP limit
    };

    // Initialize optimization settings
    this.optimizationConfig = {
      maxImageSize: options.maxImageSize || 2048,
      chunkSize: options.chunkSize || 512,
      timeout: options.timeout || 120000,
      processing: {
        timeout: options.processing?.timeout || 60000,
        retries: options.processing?.retries || 3,
        quality: options.processing?.quality || 90,
      },
      quality: options.quality || 90,
      defaultFormat: "jpeg",
      convertOptions: {
        quality: 90,
        progressive: true,
        optimizeCoding: true,
        trellisQuantisation: true,
        overshootDeringing: true,
        optimizeScans: true,
        mozjpeg: true,
      },
    };

    // Initialize analyzers with config
    this.colorAnalyzer = new ColorAnalysis(this.config.COLOR);
    this.edgeAnalyzer = new EdgeAnalysis(this.config.EDGE);
    this.textureAnalyzer = new TextureAnalysis(this.config.TEXTURE);
  }

  initializeAnalyzers() {
    this.colorAnalyzer = new ColorAnalysis();
    this.edgeAnalyzer = new EdgeAnalysis();
    this.textureAnalyzer = new TextureAnalysis();
  }

  async validateAndPreprocessBuffer(buffer) {
    try {
      // Basic validation
      if (!Buffer.isBuffer(buffer)) {
        throw new Error("Invalid input: Buffer expected");
      }

      if (buffer.length === 0) {
        throw new Error("Invalid input: Empty buffer");
      }

      // Create a defensive copy of the buffer
      const bufferCopy = Buffer.from(buffer);

      // Initial metadata check
      const metadata = await sharp(bufferCopy, {
        failOnError: false,
        unlimited: true,
        sequentialRead: true,
      }).metadata();

      if (!metadata || !metadata.width || !metadata.height) {
        throw new Error("Invalid image metadata");
      }

      // Process the image with consistent settings
      const processed = await sharp(bufferCopy, {
        failOnError: false,
        unlimited: true,
        sequentialRead: true,
      })
        .rotate() // Auto-rotate based on EXIF
        .resize(2048, 2048, {
          fit: "inside",
          withoutEnlargement: true,
          kernel: "lanczos3",
        })
        .jpeg({
          quality: 90,
          chromaSubsampling: "4:4:4",
          force: true,
          mozjpeg: true,
        })
        .toBuffer();

      // Verify processed buffer
      const processedMetadata = await sharp(processed, {
        failOnError: false,
      }).metadata();

      if (
        !processedMetadata ||
        !processedMetadata.width ||
        !processedMetadata.height
      ) {
        throw new Error("Invalid processed buffer");
      }

      return {
        buffer: processed,
        metadata: processedMetadata,
      };
    } catch (error) {
      throw new Error(`Buffer preprocessing failed: ${error.message}`);
    }
  }

  async _validateBuffer(buffer) {
    if (!Buffer.isBuffer(buffer)) {
      throw new Error("Invalid input: Not a buffer");
    }

    // Create a copy to ensure buffer integrity
    const copy = Buffer.from(buffer);

    try {
      // Test buffer with sharp
      const metadata = await sharp(copy).metadata();
      if (!metadata.width || !metadata.height) {
        throw new Error("Invalid image dimensions");
      }

      return {
        buffer: copy,
        metadata,
      };
    } catch (error) {
      throw new Error(`Buffer validation failed: ${error.message}`);
    }
  }

  async cleanupTemporaryResources() {
    try {
      // Clear any caches
      this.colorAnalyzer?.cache?.clear();
      this.edgeAnalyzer?.cache?.clear();
      this.textureAnalyzer?.cache?.clear();

      // Force garbage collection if available
      if (global.gc) {
        global.gc();
      }
    } catch (error) {
      console.warn("Resource cleanup warning:", error);
    }
  }

  // Add timeout wrapper
  withTimeout = async (promise, timeoutMs = 30000, operation = "Operation") => {
    let timeoutId;
    try {
      const result = await Promise.race([
        promise,
        new Promise((_, reject) => {
          timeoutId = setTimeout(() => {
            reject(new Error(`${operation} timed out after ${timeoutMs}ms`));
          }, timeoutMs);
        }),
      ]);
      return result;
    } finally {
      clearTimeout(timeoutId);
    }
  };

  // Optimized buffer validation and preprocessing

  async validateAndProcessImage(buffer) {
    const startTime = Date.now();
    let processedBuffer = null;

    try {
      if (!Buffer.isBuffer(buffer)) {
        throw new Error("Invalid input: Not a buffer");
      }

      // Configure sharp for optimal processing
      sharp.cache(false);
      sharp.concurrency(1);
      sharp.simd(true);

      // Configuration options
      const sharpOptions = {
        failOnError: false,
        unlimited: true,
        sequentialRead: true,
        density: 300,
      };

      const jpegOptions = {
        quality: 90,
        chromaSubsampling: "4:4:4",
        force: true,
        mozjpeg: true,
        optimizeCoding: true,
        trellisQuantisation: true,
        overshootDeringing: true,
        optimizeScans: true,
      };

      // Process the buffer to ensure JPEG format
      processedBuffer = await sharp(buffer, sharpOptions)
        .rotate() // Auto-rotate based on EXIF
        .resize(2048, 2048, {
          fit: "inside",
          withoutEnlargement: true,
          kernel: "lanczos3",
        })
        .removeAlpha() // Ensure no alpha channel
        .jpeg(jpegOptions)
        .toBuffer();

      // Verify the processed buffer
      const metadata = await sharp(processedBuffer, {
        failOnError: false,
      }).metadata();

      if (
        !metadata ||
        metadata.format !== "jpeg" ||
        !metadata.width ||
        !metadata.height
      ) {
        throw new Error("Failed to verify JPEG format");
      }

      // Additional validation checks
      if (metadata.width < 8 || metadata.height < 8) {
        throw new Error("Image dimensions too small (minimum 8x8)");
      }

      if (metadata.width > 20000 || metadata.height > 20000) {
        throw new Error("Image dimensions too large (maximum 20000x20000)");
      }

      const pixelCount = metadata.width * metadata.height;
      if (pixelCount > 40000000) {
        // 40MP limit
        throw new Error("Image pixel count exceeds maximum allowed (40MP)");
      }

      logger.info("Image validation successful:", {
        format: metadata.format,
        dimensions: `${metadata.width}x${metadata.height}`,
        size: Math.round(processedBuffer.length / 1024) + "KB",
        duration: Date.now() - startTime + "ms",
      });

      return { validBuffer: processedBuffer, metadata };
    } catch (error) {
      logger.error("Image validation failed:", {
        error: error.message,
        stack: error.stack,
        duration: Date.now() - startTime + "ms",
      });

      return { validBuffer: null, metadata: null };
    } finally {
      // Reset Sharp settings
      sharp.cache(false);
      sharp.concurrency(1);
    }
  }

  async _processBufferWithFallback(buffer) {
    const options = {
      failOnError: false,
      unlimited: true,
      sequentialRead: true,
    };

    // Try different processing strategies
    const strategies = [
      // Direct processing
      async () => {
        const processed = await sharp(buffer, options)
          .rotate()
          .ensureAlpha()
          .raw()
          .toBuffer();
        const metadata = await sharp(processed, options).metadata();
        return { buffer: processed, metadata };
      },
      // JPEG conversion
      async () => {
        const processed = await sharp(buffer, options)
          .jpeg({ quality: 90, chromaSubsampling: "4:4:4" })
          .toBuffer();
        const metadata = await sharp(processed, options).metadata();
        return { buffer: processed, metadata };
      },
      // PNG conversion
      async () => {
        const processed = await sharp(buffer, options)
          .png({ compressionLevel: 6 })
          .toBuffer();
        const metadata = await sharp(processed, options).metadata();
        return { buffer: processed, metadata };
      },
    ];

    // Try each strategy
    for (const strategy of strategies) {
      try {
        const result = await strategy();
        if (result.metadata?.width && result.metadata?.height) {
          return result;
        }
      } catch (error) {
        console.error("Processing strategy failed:", error.message);
        continue;
      }
    }

    return null;
  }

  async getMetadata(buffer, options = {}) {
    try {
      const metadata = await sharp(buffer, {
        failOnError: false,
        ...options,
      }).metadata();

      if (!metadata || !metadata.width || !metadata.height) {
        throw new Error("Invalid metadata");
      }

      return metadata;
    } catch (error) {
      throw new Error(`Failed to get metadata: ${error.message}`);
    }
  }

  isValidMetadata(metadata) {
    return (
      metadata &&
      typeof metadata.width === "number" &&
      metadata.width > 0 &&
      typeof metadata.height === "number" &&
      metadata.height > 0 &&
      metadata.width <= 20000 &&
      metadata.height <= 20000 &&
      metadata.width * metadata.height <= 40000000
    ); // 40MP limit
  }

  async processSafely(buffer, options = {}) {
    try {
      const image = sharp(buffer, {
        failOnError: false,
        unlimited: true,
        sequentialRead: true,
        ...options,
      });

      const metadata = await image.metadata();
      if (!this.isValidMetadata(metadata)) {
        throw new Error("Invalid image dimensions");
      }

      return { image, metadata };
    } catch (error) {
      throw new Error(`Image processing failed: ${error.message}`);
    }
  }

  isFormatSupported(format) {
    const supportedFormats = new Set([
      "jpeg",
      "jpg",
      "png",
      "webp",
      "tiff",
      "gif",
      "svg",
      "heic",
      "heif",
      "raw",
      "bmp",
    ]);
    return format && supportedFormats.has(format.toLowerCase());
  }

  async prepareValidBuffer(buffer) {
    try {
      if (!Buffer.isBuffer(buffer)) {
        throw new Error("Invalid input: Not a buffer");
      }

      // Create defensive copy of buffer
      const safeBuffer = Buffer.from(buffer);

      // Try to get metadata first
      let metadata;
      try {
        metadata = await sharp(safeBuffer, { failOnError: false }).metadata();
        if (metadata && this.isFormatSupported(metadata.format)) {
          return safeBuffer;
        }
      } catch (err) {
        console.log("Initial format detection failed, attempting conversion");
      }

      // Convert to supported format
      return await sharp(safeBuffer, { failOnError: false })
        .jpeg({ quality: 90 })
        .toBuffer();
    } catch (error) {
      throw new Error(`Buffer validation failed: ${error.message}`);
    }
  }

  calculateMean(values) {
    if (!values || values.length === 0) return 0;
    let sum = 0;
    for (let i = 0; i < values.length; i++) {
      sum += values[i];
    }
    return sum / values.length;
  }

  calculateStd(values) {
    if (!values || values.length === 0) return 0;
    const mean = this.calculateMean(values);
    let sumSquaredDiff = 0;
    for (let i = 0; i < values.length; i++) {
      sumSquaredDiff += Math.pow(values[i] - mean, 2);
    }
    return Math.sqrt(sumSquaredDiff / values.length);
  }

  calculateEntropy(histogram) {
    return -histogram.reduce((sum, p) => {
      if (p > 0) {
        sum += p * Math.log2(p);
      }
      return sum;
    }, 0);
  }

  popCount(n) {
    n = n - ((n >> 1) & 0x55555555);
    n = (n & 0x33333333) + ((n >> 2) & 0x33333333);
    n = (n + (n >> 4)) & 0x0f0f0f0f;
    n = n + (n >> 8);
    n = n + (n >> 16);
    return n & 0x3f;
  }

  getColorAt(colorLayout, x, y) {
    if (
      !colorLayout ||
      !colorLayout.data ||
      typeof x !== "number" ||
      typeof y !== "number" ||
      isNaN(x) ||
      isNaN(y) ||
      !colorLayout.width ||
      !colorLayout.channels
    ) {
      return {
        r: 0,
        g: 0,
        b: 0,
        a: 255,
      };
    }

    try {
      const idx = (y * colorLayout.width + x) * colorLayout.channels;
      if (idx < 0 || idx >= colorLayout.data.length) {
        return {
          r: 0,
          g: 0,
          b: 0,
          a: 255,
        };
      }

      return {
        r: colorLayout.data[idx] || 0,
        g: colorLayout.data[idx + 1] || 0,
        b: colorLayout.data[idx + 2] || 0,
        a: colorLayout.channels > 3 ? colorLayout.data[idx + 3] : 255,
      };
    } catch (error) {
      console.error("Error getting color at position:", error);
      return {
        r: 0,
        g: 0,
        b: 0,
        a: 255,
      };
    }
  }

  updateRegionBounds(bounds, x, y) {
    bounds.minX = Math.min(bounds.minX, x);
    bounds.maxX = Math.max(bounds.maxX, x);
    bounds.minY = Math.min(bounds.minY, y);
    bounds.maxY = Math.max(bounds.maxY, y);
  }

  async convertToSupportedFormat(buffer) {
    const conversionAttempts = [
      // Try WebP first (good compression, wide support)
      async () =>
        await sharp(buffer, { failOnError: false })
          .webp({ quality: 90 })
          .toBuffer(),

      // Then try JPEG with mozjpeg
      async () =>
        await sharp(buffer, { failOnError: false })
          .jpeg({
            quality: 90,
            mozjpeg: true,
            chromaSubsampling: "4:4:4",
          })
          .toBuffer(),

      // Standard JPEG as fallback
      async () =>
        await sharp(buffer, { failOnError: false })
          .jpeg({ quality: 90 })
          .toBuffer(),

      // PNG as last resort
      async () =>
        await sharp(buffer, { failOnError: false })
          .png({ compressionLevel: 9 })
          .toBuffer(),
    ];

    let lastError;
    for (const attempt of conversionAttempts) {
      try {
        const converted = await attempt();
        // Verify the converted buffer is valid
        await sharp(converted, { failOnError: false }).metadata();
        return converted;
      } catch (error) {
        lastError = error;
        console.log("Conversion attempt failed, trying next format");
      }
    }

    throw new Error(`All conversion attempts failed: ${lastError.message}`);
  }

  async _optimizeImageBuffer(buffer) {
    try {
      // First try to validate and convert the buffer
      const validBuffer = await this.prepareValidBuffer(buffer);

      // Get metadata of validated buffer
      const metadata = await sharp(validBuffer, {
        failOnError: false,
      }).metadata();

      // Resize if needed
      if (
        metadata.width > this.optimizationConfig.maxImageSize ||
        metadata.height > this.optimizationConfig.maxImageSize
      ) {
        const { width, height } = this._calculateDimensions(
          metadata.width,
          metadata.height,
          this.optimizationConfig.maxImageSize
        );

        return await sharp(validBuffer, { failOnError: false })
          .resize(width, height, {
            fit: "inside",
            withoutEnlargement: true,
            kernel: sharp.kernel.lanczos3,
          })
          .toBuffer();
      }

      return validBuffer;
    } catch (error) {
      console.error("Image optimization failed:", error);
      throw new Error(`Image optimization failed: ${error.message}`);
    }
  }

  async generateDCTHash(buffer) {
    try {
      const { data, info } = await sharp(buffer, { failOnError: false })
        .resize(this.config.HASH.DCT_SIZE, this.config.HASH.DCT_SIZE, {
          fit: "fill",
          kernel: sharp.kernel.lanczos3,
        })
        .grayscale()
        .raw()
        .toBuffer({ resolveWithObject: true })
        .catch(async (err) => {
          // Fallback preprocessing if initial attempt fails
          const processedBuffer = await this.validateAndPreprocessBuffer(
            buffer
          );
          return sharp(processedBuffer.buffer)
            .resize(this.config.HASH.DCT_SIZE, this.config.HASH.DCT_SIZE)
            .grayscale()
            .raw()
            .toBuffer({ resolveWithObject: true });
        });

      // Generate hash...
      const hash = new Uint8Array(Math.ceil(this.config.HASH.HASH_SIZE / 8));
      // Apply DCT transformation
      const dct = this.applyDCT(data, this.config.HASH.DCT_SIZE);

      // Generate binary hash
      let hashIndex = 0;
      let bitPosition = 0;

      // Use low frequency components for hash
      const medianValue = this.calculateDCTMedian(dct);

      for (let y = 0; y < 8; y++) {
        for (let x = 0; x < 8; x++) {
          if (y === 0 && x === 0) continue; // Skip DC component

          const bit =
            dct[y * this.config.HASH.DCT_SIZE + x] > medianValue ? 1 : 0;
          hash[hashIndex] |= bit << (7 - bitPosition);

          bitPosition++;
          if (bitPosition === 8) {
            bitPosition = 0;
            hashIndex++;
          }
        }
      }

      return {
        hash,
        quality: this.assessDCTHashQuality(dct, medianValue),
      };
    } catch (error) {
      console.error("DCT hash generation failed:", error);
      return {
        hash: new Uint8Array(Math.ceil(this.config.HASH.HASH_SIZE / 8)),
        quality: 0,
      };
    }
  }

  async generateWaveletHash(buffer) {
    try {
      // Use preprocessed buffer with error handling
      const { data, info } = await sharp(buffer, { failOnError: false })
        .resize(this.config.HASH.DCT_SIZE, this.config.HASH.DCT_SIZE, {
          fit: "fill",
          kernel: sharp.kernel.lanczos3,
        })
        .grayscale()
        .raw()
        .toBuffer({ resolveWithObject: true })
        .catch(async (err) => {
          // Fallback preprocessing
          const processedBuffer = await this.validateAndPreprocessBuffer(
            buffer
          );
          return sharp(processedBuffer.buffer)
            .resize(this.config.HASH.DCT_SIZE, this.config.HASH.DCT_SIZE)
            .grayscale()
            .raw()
            .toBuffer({ resolveWithObject: true });
        });

      // Apply wavelet transform
      const coefficients = this.applyHaarWavelet(
        data,
        this.config.HASH.DCT_SIZE
      );

      // Generate binary hash
      const hash = new Uint8Array(Math.ceil(this.config.HASH.HASH_SIZE / 8));
      const medianValue = this.calculateWaveletMedian(coefficients);

      let hashIndex = 0;
      let bitPosition = 0;

      for (let i = 0; i < this.config.HASH.HASH_SIZE; i++) {
        const bit = coefficients[i] > medianValue ? 1 : 0;
        hash[hashIndex] |= bit << (7 - bitPosition);

        bitPosition++;
        if (bitPosition === 8) {
          bitPosition = 0;
          hashIndex++;
        }
      }

      return {
        hash,
        quality: this.assessWaveletHashQuality(coefficients, medianValue),
      };
    } catch (error) {
      console.error("Wavelet hash generation failed:", error);
      return {
        hash: new Uint8Array(Math.ceil(this.config.HASH.HASH_SIZE / 8)),
        quality: 0,
      };
    }
  }

  async generateRadonHash(buffer) {
    try {
      // Use preprocessed buffer with error handling
      const { data, info } = await sharp(buffer, { failOnError: false })
        .resize(this.config.HASH.DCT_SIZE, this.config.HASH.DCT_SIZE, {
          fit: "fill",
          kernel: sharp.kernel.lanczos3,
        })
        .grayscale()
        .raw()
        .toBuffer({ resolveWithObject: true })
        .catch(async (err) => {
          // Fallback preprocessing
          const processedBuffer = await this.validateAndPreprocessBuffer(
            buffer
          );
          return sharp(processedBuffer.buffer)
            .resize(this.config.HASH.DCT_SIZE, this.config.HASH.DCT_SIZE)
            .grayscale()
            .raw()
            .toBuffer({ resolveWithObject: true });
        });

      // Compute Radon transform
      const radonTransform = await this.computeRadonTransform(data, info.width);
      const hash = await this.generateHashFromRadon(radonTransform);
      const quality = await this.assessRadonHashQuality(radonTransform);

      return {
        hash,
        quality,
      };
    } catch (error) {
      console.error("Radon hash generation failed:", error);
      return {
        hash: new Uint8Array(Math.ceil(this.config.HASH.HASH_SIZE / 8)),
        quality: 0,
      };
    }
  }

  async generateCombinedHash(hashes) {
    try {
      if (!hashes || !hashes.dct || !hashes.wavelet || !hashes.radon) {
        return {
          hash: new Uint8Array(Math.ceil(this.config.HASH.HASH_SIZE / 8)),
          quality: {
            robustness: 0,
            distinctiveness: 0,
            stability: 0,
          },
        };
      }

      // Ensure all hashes exist and have the hash property
      const dctHash =
        hashes.dct.hash ||
        new Uint8Array(Math.ceil(this.config.HASH.HASH_SIZE / 8));
      const waveletHash =
        hashes.wavelet.hash ||
        new Uint8Array(Math.ceil(this.config.HASH.HASH_SIZE / 8));
      const radonHash =
        hashes.radon.hash ||
        new Uint8Array(Math.ceil(this.config.HASH.HASH_SIZE / 8));

      // Combine hashes using weighted majority voting
      const combinedHash = new Uint8Array(
        Math.ceil(this.config.HASH.HASH_SIZE / 8)
      );
      const weights = [0.4, 0.3, 0.3]; // DCT, Wavelet, Radon weights

      for (let i = 0; i < combinedHash.length; i++) {
        for (let bit = 0; bit < 8; bit++) {
          let weightedSum = 0;
          weightedSum += ((dctHash[i] >> bit) & 1) * weights[0];
          weightedSum += ((waveletHash[i] >> bit) & 1) * weights[1];
          weightedSum += ((radonHash[i] >> bit) & 1) * weights[2];

          if (weightedSum >= 0.5) {
            combinedHash[i] |= 1 << bit;
          }
        }
      }

      return {
        hash: combinedHash,
        quality: this.assessCombinedHashQuality(hashes),
      };
    } catch (error) {
      console.error("Combined hash generation failed:", error);
      return {
        hash: new Uint8Array(Math.ceil(this.config.HASH.HASH_SIZE / 8)),
        quality: {
          robustness: 0,
          distinctiveness: 0,
          stability: 0,
        },
      };
    }
  }

  validateQualityMetrics(quality) {
    if (!quality) return { overall: 0, components: {} };

    const processValue = (value) => {
      if (typeof value === "number") {
        return Number.isFinite(value) ? value : 0;
      }
      if (typeof value === "object" && value !== null) {
        return Object.entries(value).reduce(
          (acc, [k, v]) => {
            acc[k] = processValue(v);
            return acc;
          },
          Array.isArray(value) ? [] : {}
        );
      }
      return value;
    };

    return processValue(quality);
  }

  getDefaultFeaturesByError(error) {
    if (error.message.includes("Buffer")) {
      return {
        error: error.message,
        features: {},
        quality: { overall: 0, components: {}, reliability: 0 },
      };
    }

    if (error.message.includes("timeout")) {
      return {
        error: error.message,
        features: {},
        quality: { overall: 0, components: {}, reliability: 0.5 },
      };
    }

    return {
      error: error.message,
      features: {},
      quality: { overall: 0, components: {}, reliability: 0 },
    };
  }

  async _safeExtract(extractFn) {
    try {
      const result = await extractFn();
      if (!result) {
        throw new Error("Extraction returned no result");
      }
      return result;
    } catch (error) {
      logger.error("Feature extraction failed:", {
        error: error.message,
        stack: error.stack,
      });
      return null;
    }
  }

  calculateGLCMQuality(glcmFeatures) {
    if (!glcmFeatures || !glcmFeatures.features) return 0;

    const {
      contrast = 0,
      correlation = 0,
      energy = 0,
      homogeneity = 0,
    } = glcmFeatures.features;

    // Normalize and combine metrics
    const normalizedContrast = Math.min(1, contrast / 255);
    const normalizedEnergy = energy; // Already normalized
    const normalizedHomogeneity = homogeneity; // Already normalized
    const normalizedCorrelation = Math.abs(correlation);

    return (
      normalizedContrast * 0.3 +
      normalizedEnergy * 0.2 +
      normalizedHomogeneity * 0.3 +
      normalizedCorrelation * 0.2
    );
  }

  calculateLBPReliability(lbpFeatures) {
    if (!lbpFeatures || !lbpFeatures.histogram) return 0;

    const histogram = lbpFeatures.histogram;
    const sum = histogram.reduce((a, b) => a + b, 0);
    if (sum === 0) return 0;

    // Calculate normalized entropy
    const entropy = histogram.reduce((e, count) => {
      const p = count / sum;
      return e - (p > 0 ? p * Math.log2(p) : 0);
    }, 0);

    // Normalize entropy to [0, 1]
    return Math.max(0, 1 - entropy / Math.log2(histogram.length));
  }

  calculateGaborReliability(gaborFeatures) {
    if (!gaborFeatures || !gaborFeatures.features) return 0;

    const responses = gaborFeatures.features.map((f) => f.magnitude || 0);
    if (responses.length === 0) return 0;

    const mean =
      responses.reduce((sum, val) => sum + val, 0) / responses.length;
    const variance =
      responses.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) /
      responses.length;

    // Higher reliability for consistent responses
    return Math.exp(-variance / (mean + 1e-6));
  }

  calculateTextureReliability(features) {
    if (!features) return 0;

    const reliabilityScores = [];

    if (features.glcm && features.glcm.features) {
      reliabilityScores.push(features.glcm.features.energy || 0);
    }

    if (features.lbp && features.lbp.histogram) {
      const lbpReliability = this.calculateLBPReliability(features.lbp);
      reliabilityScores.push(lbpReliability);
    }

    if (features.gabor && features.gabor.features) {
      const gaborReliability = this.calculateGaborReliability(features.gabor);
      reliabilityScores.push(gaborReliability);
    }

    return reliabilityScores.length > 0
      ? reliabilityScores.reduce((sum, score) => sum + score, 0) /
          reliabilityScores.length
      : 0;
  }

  calculateLBPQuality(lbpFeatures) {
    if (!lbpFeatures || !lbpFeatures.histogram) return 0;

    const histogram = lbpFeatures.histogram;

    // Calculate uniformity
    const uniformity = this.calculateLBPUniformity(histogram);

    // Calculate pattern variance
    const variance = this.calculateLBPVariance(histogram);

    // Calculate pattern entropy
    const entropy = this.calculateLBPEntropy(histogram);

    return uniformity * 0.4 + (1 - variance) * 0.3 + entropy * 0.3;
  }

  calculateGaborQuality(gaborFeatures) {
    if (
      !gaborFeatures ||
      !gaborFeatures.features ||
      !Array.isArray(gaborFeatures.features)
    ) {
      return 0;
    }

    const responses = gaborFeatures.features.map((f) => f.magnitude || 0);
    if (responses.length === 0) return 0;

    const mean =
      responses.reduce((sum, val) => sum + val, 0) / responses.length;
    const maxResponse = Math.max(...responses);

    return (mean / 255) * 0.6 + (maxResponse / 255) * 0.4;
  }

  assessTextureQuality(features) {
    if (!features)
      return {
        overall: 0,
        components: { glcm: 0, lbp: 0, gabor: 0 },
        reliability: 0,
      };

    try {
      const glcmQuality = features.glcm
        ? this.calculateGLCMQuality(features.glcm)
        : 0;

      const lbpQuality = features.lbp
        ? this.calculateLBPQuality(features.lbp)
        : 0;

      const gaborQuality = features.gabor
        ? this.calculateGaborQuality(features.gabor)
        : 0;

      const overall = glcmQuality * 0.4 + lbpQuality * 0.3 + gaborQuality * 0.3;

      const reliability = this.calculateTextureReliability(features);

      return {
        overall: Math.min(1, Math.max(0, overall)),
        components: {
          glcm: glcmQuality,
          lbp: lbpQuality,
          gabor: gaborQuality,
        },
        reliability,
      };
    } catch (error) {
      logger.error("Texture quality assessment failed:", error);
      return {
        overall: 0,
        components: { glcm: 0, lbp: 0, gabor: 0 },
        reliability: 0,
      };
    }
  }

  getDefaultTextureFeatures() {
    return {
      glcm: {
        features: {},
        statistics: {
          contrast: { mean: 0, std: 0, min: 0, max: 0, range: 0 },
          dissimilarity: { mean: 0, std: 0, min: 0, max: 0, range: 0 },
          homogeneity: { mean: 0, std: 0, min: 0, max: 0, range: 0 },
          energy: { mean: 0, std: 0, min: 0, max: 0, range: 0 },
          correlation: { mean: 0, std: 0, min: 0, max: 0, range: 0 },
          entropy: { mean: 0, std: 0, min: 0, max: 0, range: 0 },
        },
        quality: {
          overall: 0,
          components: {
            validity: 0,
            consistency: 0,
            reliability: 0,
          },
        },
      },
      lbp: {
        histogram: new Float32Array(256).fill(1 / 256),
        uniformity: 0,
        rotation: {
          invariance: 0,
          patterns: new Map(),
        },
        quality: {
          overall: 0,
          components: {
            uniformity: 0,
            stability: 0,
            distinctiveness: 0,
          },
        },
      },
      gabor: {
        features: [],
        responses: [],
        statistics: {
          mean: 0,
          std: 0,
          energy: 0,
        },
        quality: {
          overall: 0,
          reliability: 0,
        },
      },
      statistics: {
        mean: 0,
        variance: 0,
        entropy: 0,
        energy: 0,
      },
      quality: {
        overall: 0,
        components: {
          glcm: 0,
          lbp: 0,
          gabor: 0,
        },
        reliability: 0,
      },
    };
  }

  async extractTextureFeatures(buffer) {
    try {
      if (!Buffer.isBuffer(buffer)) {
        throw new Error("Invalid input: Buffer expected");
      }

      const { data, info } = await sharp(buffer, {
        failOnError: false,
        unlimited: true,
        sequentialRead: true,
      })
        .grayscale()
        .raw()
        .toBuffer({ resolveWithObject: true });

      // Extract texture features in parallel
      const [glcmFeatures, lbpFeatures, gaborFeatures] = await Promise.all([
        this._safeExtract(() => this.extractGLCMFeatures(data, info)),
        this._safeExtract(() => this.extractLBPFeatures(data, info)),
        this._safeExtract(() => this.extractGaborFeatures(data, info)),
      ]);

      const features = {
        glcm: glcmFeatures || this.getDefaultGLCMFeatures(),
        lbp: lbpFeatures || this.getDefaultLBPFeatures(),
        gabor: gaborFeatures || this.getDefaultGaborFeatures(),
        statistics: this.calculateTextureStatistics({
          glcm: glcmFeatures,
          lbp: lbpFeatures,
          gabor: gaborFeatures,
        }),
      };

      // Calculate quality metrics
      features.quality = this.assessTextureQuality(features);

      return features;
    } catch (error) {
      logger.error("Texture feature extraction failed:", {
        error: error.message,
        stack: error.stack,
      });
      return this.getDefaultTextureFeatures();
    }
  }

  async extractGLCMFeatures(data, info) {
    try {
      if (!data || !info || !info.width || !info.height) {
        return this.getDefaultGLCMFeatures();
      }

      const features = {};
      const distances = this.config.TEXTURE.GLCM_DISTANCES;
      const angles = [0, 45, 90, 135];

      for (const distance of distances) {
        features[`d${distance}`] = {};
        for (const angle of angles) {
          const glcm = await this.computeGLCM(
            data,
            info.width,
            info.height,
            distance,
            angle
          );
          const normalized = await this.normalizeGLCM(glcm);
          features[`d${distance}`][`a${angle}`] =
            this.calculateGLCMFeatures(normalized);
        }
      }

      return {
        features,
        statistics: this.calculateGLCMStatistics(features),
        quality: this.assessGLCMQuality(features),
      };
    } catch (error) {
      console.error("GLCM feature extraction failed:", error);
      return this.getDefaultGLCMFeatures();
    }
  }

  // enhanced-signature-generator.js

  async extractLBPFeatures(buffer) {
    try {
      const { data, info } = await BufferUtils.validateAndConvert(buffer);

      const rawData = await BufferUtils.preprocessForFeature(data, {
        width: info.width,
        height: info.height,
        channels: 1,
      });

      if (!rawData) {
        throw new Error("Failed to preprocess buffer for LBP");
      }

      const lbpImage = this.computeLBP(rawData, info.width, info.height);
      const histogram = this.computeLBPHistogram(lbpImage);

      if (!histogram || histogram.length === 0) {
        return this.getDefaultLBPFeatures();
      }

      const patterns = this.analyzeLBPPatterns(lbpImage);
      const rotation = this.analyzeLBPRotationInvariance(histogram);

      return {
        histogram,
        patterns,
        uniformity: this.calculateLBPUniformity(histogram),
        rotation,
        quality: this.assessLBPQuality(histogram, patterns, rotation),
      };
    } catch (error) {
      logger.error("LBP feature extraction failed:", error);
      return this.getDefaultLBPFeatures();
    }
  }

  async extractGaborFeatures(buffer) {
    try {
      const { data, info } = await BufferUtils.validateAndConvert(buffer);

      const rawData = await BufferUtils.preprocessForFeature(data, {
        width: info.width,
        height: info.height,
        channels: 1,
      });

      if (!rawData) {
        throw new Error("Failed to preprocess buffer for Gabor");
      }

      const features = [];
      const responses = [];

      // Apply Gabor filters
      for (const theta of this.config.TEXTURE.GABOR_ORIENTATIONS) {
        for (const frequency of this.config.TEXTURE.GABOR_FREQUENCIES) {
          const kernel = this.createGaborKernel(theta, frequency);
          const response = this.applyGaborFilter(
            rawData,
            info.width,
            info.height,
            kernel
          );
          responses.push(response);

          const responseFeatures = this.extractGaborResponse(response);
          features.push({
            orientation: theta,
            frequency,
            features: responseFeatures,
          });
        }
      }

      return {
        features,
        responses,
        statistics: this.calculateGaborStatistics(features),
        quality: this.assessGaborQuality(features),
      };
    } catch (error) {
      logger.error("Gabor feature extraction failed:", error);
      return this.getDefaultGaborFeatures();
    }
  }

  async extractWaveletFeatures(buffer) {
    try {
      const { data, info } = await sharp(buffer)
        .grayscale()
        .raw()
        .toBuffer({ resolveWithObject: true });

      // Ensure dimensions are power of 2 or pad
      const size = Math.pow(
        2,
        Math.ceil(Math.log2(Math.max(info.width, info.height)))
      );
      const paddedData = this.padImage(data, info.width, info.height, size);

      // Compute wavelet transform with error handling
      const coefficients = this.computeWaveletTransform(paddedData, size, size);

      // Extract statistical features safely
      const features = this.extractWaveletStatistics(coefficients);

      // Calculate energy and entropy
      const energyFeatures = this.calculateSubbandEnergy(coefficients);
      const entropyFeatures = this.calculateSubbandEntropy(coefficients);

      return {
        coefficients,
        statistics: features,
        energy: energyFeatures,
        entropy: entropyFeatures,
        quality: this.assessWaveletQuality(features),
      };
    } catch (error) {
      console.error("Wavelet feature extraction failed:", error);
      return this.getDefaultWaveletFeatures();
    }
  }

  applyNonMaxSuppression(magnitude, direction, width) {
    const suppressed = new Float32Array(magnitude.length);

    for (let y = 1; y < width - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const idx = y * width + x;
        const theta = (direction[idx] * 180) / Math.PI;
        const angle = (theta + 180) % 180;

        // Get neighbors based on gradient direction
        let neighbor1, neighbor2;

        if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle < 180)) {
          neighbor1 = magnitude[idx - 1];
          neighbor2 = magnitude[idx + 1];
        } else if (angle >= 22.5 && angle < 67.5) {
          neighbor1 = magnitude[idx - width - 1];
          neighbor2 = magnitude[idx + width + 1];
        } else if (angle >= 67.5 && angle < 112.5) {
          neighbor1 = magnitude[idx - width];
          neighbor2 = magnitude[idx + width];
        } else {
          neighbor1 = magnitude[idx - width + 1];
          neighbor2 = magnitude[idx + width - 1];
        }

        if (magnitude[idx] >= neighbor1 && magnitude[idx] >= neighbor2) {
          suppressed[idx] = magnitude[idx];
        }
      }
    }

    return suppressed;
  }

  performHysteresisThresholding(
    suppressed,
    width,
    lowThreshold,
    highThreshold
  ) {
    const height = Math.floor(suppressed.length / width);
    const result = new Float32Array(suppressed.length);
    const stack = [];

    // First pass: mark strong edges and collect them
    for (let i = 0; i < suppressed.length; i++) {
      if (suppressed[i] >= highThreshold) {
        result[i] = 255;
        stack.push(i);
      } else if (suppressed[i] >= lowThreshold) {
        result[i] = 128; // Weak edge
      }
    }

    // Second pass: trace weak edges connected to strong edges
    const neighbors = [
      -width - 1,
      -width,
      -width + 1,
      -1,
      1,
      width - 1,
      width,
      width + 1,
    ];

    while (stack.length > 0) {
      const pixel = stack.pop();
      const x = pixel % width;

      for (const offset of neighbors) {
        const neighbor = pixel + offset;
        const nx = neighbor % width;

        // Check bounds and connectivity
        if (
          neighbor >= 0 &&
          neighbor < result.length &&
          Math.abs(nx - x) <= 1 && // Ensure we don't wrap around edges
          result[neighbor] === 128
        ) {
          // Only process weak edges
          result[neighbor] = 255; // Convert to strong edge
          stack.push(neighbor);
        }
      }
    }

    // Clear remaining weak edges
    for (let i = 0; i < result.length; i++) {
      if (result[i] === 128) {
        result[i] = 0;
      }
    }

    return result;
  }

  assessEdgeQuality(features) {
    if (!features)
      return {
        overall: 0,
        components: {
          strength: 0,
          continuity: 0,
          distribution: 0,
          junctions: 0,
        },
      };

    try {
      const strengthQuality = features.strength || 0;
      const continuityQuality = features.continuity
        ? features.continuity.score || 0
        : 0;
      const distributionQuality = features.distribution
        ? features.distribution.uniformity || 0
        : 0;
      const junctionQuality = features.junctions
        ? features.junctions.density || 0
        : 0;

      const overall =
        strengthQuality * 0.3 +
        continuityQuality * 0.3 +
        distributionQuality * 0.2 +
        junctionQuality * 0.2;

      return {
        overall: Math.min(1, Math.max(0, overall)),
        components: {
          strength: strengthQuality,
          continuity: continuityQuality,
          distribution: distributionQuality,
          junctions: junctionQuality,
        },
      };
    } catch (error) {
      logger.error("Edge quality assessment failed:", error);
      return {
        overall: 0,
        components: {
          strength: 0,
          continuity: 0,
          distribution: 0,
          junctions: 0,
        },
      };
    }
  }

  getDefaultEdgeFeatures() {
    return {
      edges: new Float32Array(0),
      features: {
        strength: 0,
        orientation: {
          histogram: new Float32Array(36).fill(0),
          dominant: [],
          coherence: 0,
        },
        continuity: {
          score: 0,
          breaks: [],
          continuityMap: new Float32Array(0),
        },
        distribution: {
          density: 0,
          uniformity: 0,
        },
        junctions: {
          points: [],
          density: 0,
          distribution: {},
        },
      },
      quality: {
        overall: 0,
        components: {
          strength: 0,
          continuity: 0,
          distribution: 0,
          junctions: 0,
        },
      },
    };
  }

  ensureJpegBuffer(buffer) {
    return new Promise(async (resolve, reject) => {
      try {
        if (!Buffer.isBuffer(buffer)) {
          throw new Error("Invalid input: Buffer expected");
        }

        const processedBuffer = await sharp(buffer, {
          failOnError: false,
          unlimited: true,
          sequentialRead: true,
        })
          .jpeg({
            quality: 90,
            chromaSubsampling: "4:4:4",
            force: true,
          })
          .toBuffer()
          .catch(async () => {
            // Fallback to PNG if JPEG fails
            return await sharp(buffer, {
              failOnError: false,
              unlimited: true,
              sequentialRead: true,
            })
              .png()
              .toBuffer();
          });

        // Verify the processed buffer
        const metadata = await sharp(processedBuffer, {
          failOnError: false,
        }).metadata();

        if (!metadata || !metadata.width || !metadata.height) {
          throw new Error("Invalid processed buffer metadata");
        }

        resolve(processedBuffer);
      } catch (error) {
        reject(new Error(`Buffer conversion failed: ${error.message}`));
      }
    });
  }

  async extractEdgeFeatures(buffer) {
    try {
      const { data, info } = await sharp(buffer)
        .grayscale()
        .raw()
        .toBuffer({ resolveWithObject: true });

      if (!data || !info.width || !info.height) {
        return this.getDefaultEdgeFeatures();
      }

      // Apply edge detection
      const edges = await this.applyCannyEdgeDetection(data, info.width);

      return {
        edges,
        features: {
          strength: this.calculateEdgeStrength(edges),
          orientation: this.analyzeEdgeOrientations(edges, info.width),
          continuity: await this.analyzeEdgeContinuity(edges, info.width),
          distribution: this.analyzeEdgeDistribution(
            edges,
            info.width,
            info.height
          ),
        },
        quality: {
          overall: this.assessEdgeQuality(edges),
          components: {
            strength: this.calculateEdgeStrength(edges),
            continuity: this.analyzeEdgeContinuity(edges, info.width).score,
            distribution: this.analyzeEdgeDistribution(
              edges,
              info.width,
              info.height
            ).uniformity,
          },
        },
      };
    } catch (error) {
      logger.error("Edge feature extraction failed:", error);
      return this.getDefaultEdgeFeatures();
    }
  }

  async extractEdgeMap(buffer) {
    try {
      const { data, info } = await sharp(buffer, {
        failOnError: false,
        unlimited: true,
        sequentialRead: true,
      })
        .grayscale()
        .raw()
        .toBuffer({ resolveWithObject: true })
        .catch(async (err) => {
          // Try processing the buffer
          const processed = await this._processBufferSafely(buffer);
          if (!processed) return { data: null, info: null };

          return sharp(processed, { failOnError: false })
            .grayscale()
            .raw()
            .toBuffer({ resolveWithObject: true });
        });

      if (!data || !info) {
        return this.getDefaultEdgeFeatures();
      }

      const edges = await this.applyCannyEdgeDetection(data, info.width);
      const features = await this.extractEdgeFeatures(
        edges,
        info.width,
        info.height
      );

      return {
        edges,
        width: info.width,
        height: info.height,
        features,
        quality: this.assessEdgeQuality(edges),
      };
    } catch (error) {
      console.error("Edge map extraction failed:", error);
      return this.getDefaultEdgeFeatures();
    }
  }

  assessSpatialQuality(features) {
    if (!features) {
      return {
        overall: 0,
        components: {
          regions: 0,
          relationships: 0,
          hierarchy: 0,
        },
      };
    }

    try {
      const regionQuality = this.assessRegionQuality(features.regions);
      const relationshipQuality = this.assessRelationshipQuality(
        features.relationships
      );
      const hierarchyQuality = this.assessHierarchyQuality(features.hierarchy);

      const overall =
        regionQuality * 0.4 +
        relationshipQuality * 0.3 +
        hierarchyQuality * 0.3;

      return {
        overall: Math.min(1, Math.max(0, overall)),
        components: {
          regions: regionQuality,
          relationships: relationshipQuality,
          hierarchy: hierarchyQuality,
        },
      };
    } catch (error) {
      logger.error("Spatial quality assessment failed:", error);
      return {
        overall: 0,
        components: {
          regions: 0,
          relationships: 0,
          hierarchy: 0,
        },
      };
    }
  }

  calculateGridCoverage(grid) {
    if (!grid || !grid.length) return 0;
    let coverage = 0;
    let total = 0;

    grid.forEach((row) => {
      row.forEach((cell) => {
        if (cell.features && cell.features.importance > 0) {
          coverage += cell.features.importance;
        }
        total++;
      });
    });

    return total > 0 ? coverage / total : 0;
  }

  calculateVariance(values, mean) {
    if (!Array.isArray(values) || values.length === 0) return 0;
    mean = mean || this.calculateMean(values);
    return (
      values.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) /
      values.length
    );
  }

  calculateGridUniformity(grid) {
    if (!grid || !grid.length) return 0;

    const importances = [];
    grid.forEach((row) => {
      row.forEach((cell) => {
        if (cell.features) {
          importances.push(cell.features.importance || 0);
        }
      });
    });

    if (importances.length === 0) return 0;

    const mean = mathUtils.calculateMean(importances);
    const variance = mathUtils.calculateVariance(importances, mean);

    return Math.exp(-variance * 2); // Higher uniformity = lower variance
  }

  calculateGridConsistency(grid) {
    if (!grid || !grid.length) return 0;

    let consistency = 0;
    let count = 0;

    grid.forEach((row) => {
      row.forEach((cell) => {
        if (cell.features) {
          consistency += cell.features.consistency || 0;
          count++;
        }
      });
    });

    return count > 0 ? consistency / count : 0;
  }

  assessGridQuality(grid) {
    if (!grid) return 0;

    const coverageScore = this.calculateGridCoverage(grid);
    const uniformityScore = this.calculateGridUniformity(grid);
    const consistencyScore = this.calculateGridConsistency(grid);

    return coverageScore * 0.4 + uniformityScore * 0.3 + consistencyScore * 0.3;
  }

  calculateSignificanceByType(typeRegions) {
    if (!typeRegions || !typeRegions.regions) return 0;

    return (
      typeRegions.regions.reduce((significance, region) => {
        const size = region.statistics?.size || 0;
        const contrast = region.features?.contrast || 0;
        const complexity = region.features?.complexity || 0;

        return significance + (size * 0.4 + contrast * 0.3 + complexity * 0.3);
      }, 0) / Math.max(1, typeRegions.regions.length)
    );
  }

  calculateRegionSignificance(regions) {
    if (!regions) return 0;

    const colorSignificance = this.calculateSignificanceByType(regions.color);
    const edgeSignificance = this.calculateSignificanceByType(regions.edge);
    const textureSignificance = this.calculateSignificanceByType(
      regions.texture
    );

    return {
      color: colorSignificance,
      edge: edgeSignificance,
      texture: textureSignificance,
      overall: (colorSignificance + edgeSignificance + textureSignificance) / 3,
    };
  }

  calculateBounds(points) {
    if (!points || !Array.isArray(points) || points.length === 0) {
      return { minX: 0, maxX: 0, minY: 0, maxY: 0 };
    }

    return points.reduce(
      (bounds, [x, y]) => ({
        minX: Math.min(bounds.minX, x),
        maxX: Math.max(bounds.maxX, x),
        minY: Math.min(bounds.minY, y),
        maxY: Math.max(bounds.maxY, y),
      }),
      {
        minX: points[0][0],
        maxX: points[0][0],
        minY: points[0][1],
        maxY: points[0][1],
      }
    );
  }

  analyzeRegionShape(region) {
    if (!region || !region.points || !Array.isArray(region.points)) {
      return {
        circularity: 0,
        elongation: 0,
        orientation: 0,
      };
    }

    const points = region.points;
    const perimeter = this.calculatePerimeter(points);
    const area = points.length;

    // Calculate circularity (1 for perfect circle)
    const circularity = (4 * Math.PI * area) / (perimeter * perimeter);

    // Calculate elongation
    const bounds = this.calculateBounds(points);
    const width = bounds.maxX - bounds.minX + 1;
    const height = bounds.maxY - bounds.minY + 1;
    const elongation = Math.max(width / height, height / width);

    // Calculate orientation
    const orientation = this.calculateRegionOrientation(region);

    return {
      circularity: Math.min(1, circularity),
      elongation: Math.min(1, elongation / 4), // Normalize elongation
      orientation: orientation,
    };
  }

  calculateRegionCoherence(regions) {
    if (!regions || !Array.isArray(regions) || regions.length === 0) {
      return 0;
    }

    const coherenceScores = regions.map((region) => {
      if (!region || !region.points || !Array.isArray(region.points)) {
        return 0;
      }

      const shape = this.analyzeRegionShape(region);
      const density = this.calculateRegionDensity(region);
      const compactness = this.calculateRegionCompactness(region);

      return shape.circularity * 0.3 + density * 0.4 + compactness * 0.3;
    });

    return (
      coherenceScores.reduce((sum, score) => sum + score, 0) /
      coherenceScores.length
    );
  }

  calculateRegionStability(regions) {
    if (!regions || !Array.isArray(regions) || regions.length === 0) {
      return 0;
    }

    try {
      let totalStability = 0;
      let validRegions = 0;

      for (const region of regions) {
        if (!region || !region.points || !Array.isArray(region.points)) {
          continue;
        }

        // Calculate perimeter to area ratio for stability
        const perimeter = this.calculatePerimeter(region.points);
        const area = region.points.length;
        if (area === 0 || perimeter === 0) continue;

        // Calculate shape stability metrics
        const compactness = (4 * Math.PI * area) / (perimeter * perimeter);
        const elongation = this.calculateRegionElongation(region);
        const density = this.calculateRegionDensity(region);

        // Combined stability metric
        const stability =
          compactness * 0.4 + (1 - elongation) * 0.3 + density * 0.3;

        totalStability += stability;
        validRegions++;
      }

      return validRegions > 0 ? totalStability / validRegions : 0;
    } catch (error) {
      logger.error("Error calculating region stability:", error);
      return 0;
    }
  }

  assessRegionQuality(regions) {
    if (!regions) return 0;

    const significanceScore = this.calculateRegionSignificance(regions);
    const coherenceScore = this.calculateRegionCoherence(regions);
    const stabilityScore = this.calculateRegionStability(regions);

    return (
      significanceScore * 0.4 + coherenceScore * 0.3 + stabilityScore * 0.3
    );
  }

  calculateRelationshipStrength(relationships) {
    if (
      !relationships ||
      !Array.isArray(relationships) ||
      relationships.length === 0
    ) {
      return 0;
    }

    try {
      let totalStrength = 0;
      let validRelationships = 0;

      for (const relationship of relationships) {
        if (!relationship || typeof relationship.strength !== "number") {
          continue;
        }

        totalStrength += relationship.strength;
        validRelationships++;
      }

      return validRelationships > 0 ? totalStrength / validRelationships : 0;
    } catch (error) {
      logger.error("Error calculating relationship strength:", error);
      return 0;
    }
  }

  assessRelationshipQuality(relationships) {
    if (!relationships) return 0;

    const strengthScore = this.calculateRelationshipStrength(relationships);
    const consistencyScore =
      this.calculateRelationshipConsistency(relationships);
    const reliabilityScore =
      this.calculateRelationshipReliability(relationships);

    return (
      strengthScore * 0.4 + consistencyScore * 0.3 + reliabilityScore * 0.3
    );
  }

  isSignificantRegion(region) {
    if (!region || !Array.isArray(region.points)) {
      return false;
    }

    return region.points.length >= 64; // Minimum size threshold
  }

  async segmentColorRegions(colorLayout) {
    if (!colorLayout || !colorLayout.width || !colorLayout.height) {
      return {
        regions: [],
        statistics: this.calculateRegionStatistics([]),
        relationships: [],
      };
    }

    try {
      const regions = [];
      const visited = new Set();

      // Segment regions using color similarity
      for (let y = 0; y < colorLayout.height; y++) {
        for (let x = 0; x < colorLayout.width; x++) {
          const key = `${x},${y}`;
          if (!visited.has(key)) {
            const region = await this.growColorRegion(
              colorLayout,
              x,
              y,
              visited
            );
            if (this.isSignificantRegion(region)) {
              regions.push(this.analyzeColorRegion(region));
            }
          }
        }
      }

      return {
        regions,
        statistics: this.calculateRegionStatistics(regions),
        relationships: this.analyzeRegionRelationships(regions),
      };
    } catch (error) {
      logger.error("Color region segmentation failed:", error);
      return {
        regions: [],
        statistics: this.calculateRegionStatistics([]),
        relationships: [],
      };
    }
  }

  traceEdgeRegion(edges, startX, startY, width, visited) {
    const region = {
      points: [],
      strength: 0,
      bounds: {
        minX: startX,
        maxX: startX,
        minY: startY,
        maxY: startY,
      },
    };

    const stack = [[startX, startY]];
    while (stack.length > 0) {
      const [x, y] = stack.pop();
      const idx = y * width + x;

      if (visited.has(idx)) continue;
      visited.add(idx);

      region.points.push([x, y]);
      region.strength += edges[idx];
      this.updateRegionBounds(region.bounds, x, y);

      // Check 8-connected neighbors
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue;

          const nx = x + dx;
          const ny = y + dy;
          const nidx = ny * width + nx;

          if (edges[nidx] > 0 && !visited.has(nidx)) {
            stack.push([nx, ny]);
          }
        }
      }
    }

    return region;
  }

  analyzeEdgeRegion(region) {
    return {
      ...region,
      features: {
        orientation: this.calculateRegionOrientation(region),
        elongation: this.calculateRegionElongation(region),
        circularity: this.calculateRegionCircularity(region),
        convexity: this.calculateRegionConvexity(region),
      },
    };
  }

  segmentEdgeRegions(edgeMap) {
    if (!edgeMap || !edgeMap.edges) {
      return [];
    }

    const regions = [];
    const visited = new Set();
    const width = edgeMap.width;
    const height = Math.floor(edgeMap.edges.length / width);

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = y * width + x;
        if (edgeMap.edges[idx] > 0 && !visited.has(idx)) {
          const region = this.traceEdgeRegion(
            edgeMap.edges,
            x,
            y,
            width,
            visited
          );
          if (this.isSignificantRegion(region)) {
            regions.push(this.analyzeEdgeRegion(region));
          }
        }
      }
    }

    return regions;
  }

  async getTextureFeatures(features, x, y) {
    try {
      const glcmFeatures = features.glcm?.[y]?.[x] || {};
      const lbpFeatures = features.lbp?.[y]?.[x] || {};
      const gaborFeatures = features.gabor?.[y]?.[x] || {};

      return {
        contrast: glcmFeatures.contrast || 0,
        homogeneity: glcmFeatures.homogeneity || 0,
        correlation: glcmFeatures.correlation || 0,
        lbpValue: lbpFeatures.value || 0,
        gaborResponse: gaborFeatures.magnitude || 0,
      };
    } catch (error) {
      return {
        contrast: 0,
        homogeneity: 0,
        correlation: 0,
        lbpValue: 0,
        gaborResponse: 0,
      };
    }
  }

  isTextureSimilar(texture1, texture2, threshold) {
    const differences = [
      Math.abs(texture1.contrast - texture2.contrast),
      Math.abs(texture1.homogeneity - texture2.homogeneity),
      Math.abs(texture1.correlation - texture2.correlation),
      Math.abs(texture1.lbpValue - texture2.lbpValue),
      Math.abs(texture1.gaborResponse - texture2.gaborResponse),
    ];

    const avgDifference =
      differences.reduce((a, b) => a + b, 0) / differences.length;
    return avgDifference <= threshold;
  }

  calculateTextureThreshold(texture) {
    // Adaptive threshold based on texture characteristics
    const baseThreshold = 0.3;
    const variabilityFactor = (texture.contrast + texture.gaborResponse) / 2;
    return baseThreshold * (1 + variabilityFactor);
  }

  async growTextureRegion(features, startX, startY, visited) {
    if (!features || !features.glcm) {
      return {
        points: [],
        features: [],
        bounds: { minX: 0, maxX: 0, minY: 0, maxY: 0 },
      };
    }

    const region = {
      points: [],
      features: [],
      bounds: {
        minX: startX,
        maxX: startX,
        minY: startY,
        maxY: startY,
      },
    };

    const stack = [[startX, startY]];
    const baseTexture = await this.getTextureFeatures(features, startX, startY);
    const threshold = this.calculateTextureThreshold(baseTexture);

    while (stack.length > 0) {
      const [x, y] = stack.pop();
      const key = `${x},${y}`;

      if (!visited.has(key)) {
        visited.add(key);
        const currentTexture = await this.getTextureFeatures(features, x, y);

        if (this.isTextureSimilar(baseTexture, currentTexture, threshold)) {
          region.points.push([x, y]);
          region.features.push(currentTexture);
          this.updateRegionBounds(region.bounds, x, y);

          // Add valid neighbors to stack
          this.addValidNeighbors(
            stack,
            x,
            y,
            features.metadata?.width || 256,
            features.metadata?.height || 256,
            visited
          );
        }
      }
    }

    return region;
  }

  async segmentTextureRegions(textureFeatures) {
    if (!textureFeatures || !textureFeatures.glcm) {
      return [];
    }

    try {
      const regions = [];
      const visited = new Set();
      const width = textureFeatures.metadata?.width || 256;
      const height = textureFeatures.metadata?.height || 256;

      // Use GLCM features to detect texture boundaries
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const key = `${x},${y}`;
          if (!visited.has(key)) {
            const region = await this.growTextureRegion(
              textureFeatures,
              x,
              y,
              visited
            );
            if (this.isSignificantRegion(region)) {
              const analyzedRegion = this.analyzeTextureRegion(region);
              regions.push(analyzedRegion);
            }
          }
        }
      }

      return {
        regions,
        statistics: this.calculateRegionStatistics(regions),
        relationships: this.analyzeRegionRelationships(regions),
      };
    } catch (error) {
      logger.error("Texture region segmentation failed:", error);
      return {
        regions: [],
        statistics: this.getDefaultRegionStatistics(),
        relationships: [],
      };
    }
  }

  buildRegionHierarchy(regions) {
    const hierarchy = {
      levels: [],
      relations: [],
      structure: {},
    };

    // Build hierarchy levels
    hierarchy.levels = this.constructHierarchicalLevels(regions);

    // Analyze relationships between levels
    hierarchy.relations = this.analyzeHierarchicalRelations(regions);

    // Build hierarchical structure
    hierarchy.structure = this.analyzeHierarchicalStructure(regions);

    return hierarchy;
  }

  async analyzeSpatialRegions(colorLayout, edgeMap, textureRegions) {
    try {
      // Extract regions for each feature type
      const regions = {
        color: await this.segmentColorRegions(colorLayout),
        edge: this.segmentEdgeRegions(edgeMap),
        texture: await this.segmentTextureRegions(textureRegions),
      };

      // Analyze region hierarchies and relationships
      const hierarchy = this.buildRegionHierarchy(regions);
      const relationships = this.analyzeRegionRelationships(regions);
      const significance = this.calculateRegionSignificance(regions);

      return {
        regions,
        hierarchy,
        relationships,
        significance,
      };
    } catch (error) {
      logger.error("Spatial region analysis failed:", error);
      return {
        regions: [],
        hierarchy: [],
        relationships: [],
        significance: 0,
      };
    }
  }

  calculateRegionDistance(region1, region2) {
    // Calculate center points
    const center1 = this.calculateRegionCenter(region1);
    const center2 = this.calculateRegionCenter(region2);

    // Calculate Euclidean distance between centers
    return Math.sqrt(
      Math.pow(center2.x - center1.x, 2) + Math.pow(center2.y - center1.y, 2)
    );
  }

  calculateRegionOverlap(region1, region2) {
    const bounds1 = region1.bounds;
    const bounds2 = region2.bounds;

    // Calculate intersection area
    const xOverlap = Math.max(
      0,
      Math.min(bounds1.maxX, bounds2.maxX) -
        Math.max(bounds1.minX, bounds2.minX)
    );

    const yOverlap = Math.max(
      0,
      Math.min(bounds1.maxY, bounds2.maxY) -
        Math.max(bounds1.minY, bounds2.minY)
    );

    const overlapArea = xOverlap * yOverlap;

    // Calculate union area
    const area1 = (bounds1.maxX - bounds1.minX) * (bounds1.maxY - bounds1.minY);
    const area2 = (bounds2.maxX - bounds2.minX) * (bounds2.maxY - bounds2.minY);
    const unionArea = area1 + area2 - overlapArea;

    return unionArea > 0 ? overlapArea / unionArea : 0;
  }

  calculateRegionAlignment(region1, region2) {
    const center1 = this.calculateRegionCenter(region1);
    const center2 = this.calculateRegionCenter(region2);

    // Calculate horizontal and vertical alignment
    const horizontalAlignment =
      Math.abs(center1.y - center2.y) / Math.abs(center1.x - center2.x + 1e-6);
    const verticalAlignment =
      Math.abs(center1.x - center2.x) / Math.abs(center1.y - center2.y + 1e-6);

    return Math.min(horizontalAlignment, verticalAlignment);
  }

  calculateRegionSimilarity(region1, region2) {
    const colorSimilarity = this.calculateColorSimilarity(
      region1.color?.dominant,
      region2.color?.dominant
    );

    const textureSimilarity = this.calculateTextureSimilarity(
      region1.texture,
      region2.texture
    );

    const shapeSimilarity = this.calculateShapeSimilarity(
      region1.shape,
      region2.shape
    );

    return {
      color: colorSimilarity,
      texture: textureSimilarity,
      shape: shapeSimilarity,
      overall: (colorSimilarity + textureSimilarity + shapeSimilarity) / 3,
    };
  }

  calculateRelationshipFeatures(region1, region2) {
    return {
      overlap: this.calculateRegionOverlap(region1, region2),
      distance: this.calculateRegionDistance(region1, region2),
      alignment: this.calculateRegionAlignment(region1, region2),
      similarity: this.calculateRegionSimilarity(region1, region2),
    };
  }

  determineRelationshipType(distance, overlap, alignment) {
    if (overlap > 0.5) return "overlapping";
    if (overlap > 0) return "intersecting";
    if (alignment < 0.2) return "aligned";
    if (distance < 50) return "adjacent";
    return "separated";
  }

  calculateSpatialMetrics(region1, region2) {
    const distance = this.calculateRegionDistance(region1, region2);
    const overlap = this.calculateRegionOverlap(region1, region2);
    const alignment = this.calculateRegionAlignment(region1, region2);

    return {
      distance,
      overlap,
      alignment,
      relationship: this.determineRelationshipType(
        distance,
        overlap,
        alignment
      ),
    };
  }

  calculateRegionRelationships(regions) {
    const relationships = [];

    // Analyze relationships between all region pairs
    for (let i = 0; i < regions.length; i++) {
      for (let j = i + 1; j < regions.length; j++) {
        relationships.push({
          type: "spatial",
          regions: [i, j],
          metrics: this.calculateSpatialMetrics(regions[i], regions[j]),
          features: this.calculateRelationshipFeatures(regions[i], regions[j]),
        });
      }
    }

    return relationships;
  }

  analyzeSpatialRelationships(colorLayout, edgeMap, textureRegions) {
    const relationships = [];

    if (!colorLayout || !edgeMap || !textureRegions) {
      return relationships;
    }

    try {
      // Extract regions from each feature type
      const colorRegions = this.segmentColorRegions(colorLayout);
      const edgeRegions = this.segmentEdgeRegions(edgeMap);
      const textureRegions = this.extractTextureRegions(textureRegions);

      // Analyze relationships between regions
      for (let i = 0; i < colorRegions.length; i++) {
        for (let j = i + 1; j < colorRegions.length; j++) {
          relationships.push({
            type: "color-color",
            region1: i,
            region2: j,
            relationship: this.calculateRegionRelationship(
              colorRegions[i],
              colorRegions[j]
            ),
          });
        }
      }

      // Add cross-feature relationships
      colorRegions.forEach((colorRegion, i) => {
        edgeRegions.forEach((edgeRegion, j) => {
          relationships.push({
            type: "color-edge",
            region1: i,
            region2: j,
            relationship: this.calculateCrossFeatureRelationship(
              colorRegion,
              edgeRegion
            ),
          });
        });
      });

      return relationships;
    } catch (error) {
      console.error("Error analyzing spatial relationships:", error);
      return relationships;
    }
  }

  getDefaultSpatialFeatures() {
    return {
      grid: [],
      regions: [],
      relationships: [],
      invariants: {},
      metadata: {
        width: 0,
        height: 0,
        scale: 1,
      },
      quality: {
        overall: 0,
        components: {
          grid: 0,
          regions: 0,
          relationships: 0,
        },
      },
    };
  }

  detectHorizontalSymmetry(colorLayout, edgeMap) {
    if (!colorLayout || !edgeMap) return 0;

    const width = colorLayout.width;
    const height = colorLayout.height;
    let symmetryScore = 0;
    let totalComparisons = 0;

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width / 2; x++) {
        const leftColor = this.getColorAt(colorLayout, x, y);
        const rightColor = this.getColorAt(colorLayout, width - 1 - x, y);

        const leftEdge = edgeMap.edges[y * width + x];
        const rightEdge = edgeMap.edges[y * width + (width - 1 - x)];

        const colorSimilarity = this.calculateColorSimilarity(
          leftColor,
          rightColor
        );
        const edgeSimilarity = Math.abs(leftEdge - rightEdge) < 0.1 ? 1 : 0;

        symmetryScore += (colorSimilarity + edgeSimilarity) / 2;
        totalComparisons++;
      }
    }

    return totalComparisons > 0 ? symmetryScore / totalComparisons : 0;
  }

  detectVerticalSymmetry(colorLayout, edgeMap) {
    if (!colorLayout || !edgeMap) return 0;

    const width = colorLayout.width;
    const height = colorLayout.height;
    let symmetryScore = 0;
    let totalComparisons = 0;

    for (let x = 0; x < width; x++) {
      for (let y = 0; y < height / 2; y++) {
        const topColor = this.getColorAt(colorLayout, x, y);
        const bottomColor = this.getColorAt(colorLayout, x, height - 1 - y);

        const topEdge = edgeMap.edges[y * width + x];
        const bottomEdge = edgeMap.edges[(height - 1 - y) * width + x];

        const colorSimilarity = this.calculateColorSimilarity(
          topColor,
          bottomColor
        );
        const edgeSimilarity = Math.abs(topEdge - bottomEdge) < 0.1 ? 1 : 0;

        symmetryScore += (colorSimilarity + edgeSimilarity) / 2;
        totalComparisons++;
      }
    }

    return totalComparisons > 0 ? symmetryScore / totalComparisons : 0;
  }

  detectRotationalSymmetry(colorLayout, edgeMap) {
    if (!colorLayout || !edgeMap) return 0;

    const width = colorLayout.width;
    const height = colorLayout.height;
    const centerX = width / 2;
    const centerY = height / 2;

    let symmetryScore = 0;
    let totalComparisons = 0;

    // Check for 180-degree rotational symmetry
    for (let y = 0; y < height / 2; y++) {
      for (let x = 0; x < width / 2; x++) {
        const color1 = this.getColorAt(colorLayout, x, y);
        const color2 = this.getColorAt(
          colorLayout,
          width - 1 - x,
          height - 1 - y
        );

        const edge1 = edgeMap.edges[y * width + x];
        const edge2 = edgeMap.edges[(height - 1 - y) * width + (width - 1 - x)];

        const colorSimilarity = this.calculateColorSimilarity(color1, color2);
        const edgeSimilarity = Math.abs(edge1 - edge2) < 0.1 ? 1 : 0;

        symmetryScore += (colorSimilarity + edgeSimilarity) / 2;
        totalComparisons++;
      }
    }

    return totalComparisons > 0 ? symmetryScore / totalComparisons : 0;
  }

  calculateSymmetryInvariants(colorLayout, edgeMap) {
    const horizontal = this.detectHorizontalSymmetry(colorLayout, edgeMap);
    const vertical = this.detectVerticalSymmetry(colorLayout, edgeMap);
    const rotational = this.detectRotationalSymmetry(colorLayout, edgeMap);

    return {
      horizontal,
      vertical,
      rotational,
      overall: (horizontal + vertical + rotational) / 3,
    };
  }

  calculateSpatialInvariants(colorLayout, edgeMap, textureRegions) {
    return {
      symmetry: this.calculateSymmetryInvariants(colorLayout, edgeMap),
      scale: this.calculateScaleInvariants(
        colorLayout,
        edgeMap,
        textureRegions
      ),
      rotation: this.calculateRotationInvariants(
        colorLayout,
        edgeMap,
        textureRegions
      ),
      translation: this.calculateTranslationInvariants(
        colorLayout,
        edgeMap,
        textureRegions
      ),
    };
  }

  async generateSpatialVerification(buffer) {
    try {
      if (!Buffer.isBuffer(buffer)) {
        throw new Error("Invalid input: Buffer expected");
      }

      const { data, info } = await sharp(buffer, {
        failOnError: false,
        unlimited: true,
        sequentialRead: true,
      })
        .raw()
        .toBuffer({ resolveWithObject: true });

      // Extract base features in parallel
      const [colorLayout, edgeMap, textureFeatures] = await Promise.all([
        this.extractColorLayout(data, info),
        this.extractEdgeMap(data, info),
        this.extractTextureRegions(data, info),
      ]);

      // Generate spatial features
      const spatialFeatures = {
        grid: await this.generateSpatialGrid(
          data,
          info,
          colorLayout,
          edgeMap,
          textureFeatures
        ),
        regions: await this.analyzeSpatialRegions(
          colorLayout,
          edgeMap,
          textureFeatures
        ),
        relationships: this.analyzeSpatialRelationships(
          colorLayout,
          edgeMap,
          textureFeatures
        ),
        invariants: this.calculateSpatialInvariants(
          colorLayout,
          edgeMap,
          textureFeatures
        ),
      };

      // Calculate quality metrics
      spatialFeatures.quality = this.assessSpatialQuality(spatialFeatures);

      return spatialFeatures;
    } catch (error) {
      logger.error("Spatial verification failed:", error);
      return this.getDefaultSpatialFeatures();
    }
  }

  async extractColorLayout(data, info) {
    try {
      const gridSize = 8;
      const layout = {
        width: info.width,
        height: info.height,
        grid: Array(gridSize)
          .fill()
          .map(() => Array(gridSize).fill(null)),
        dominantColors: [],
        colorDistribution: new Map(),
      };

      // Process each grid cell
      for (let y = 0; y < gridSize; y++) {
        for (let x = 0; x < gridSize; x++) {
          const cellColors = this.extractCellColors(data, info, x, y, gridSize);
          layout.grid[y][x] = {
            dominantColor: await this.findDominantColor(cellColors),
            colorVariance: this.calculateCellColorVariance(cellColors),
            edgeStrength: this.calculateCellEdgeStrength(cellColors),
          };
        }
      }

      // Extract global color information
      layout.dominantColors = await this.extractDominantColors(data, info);
      layout.colorDistribution = this.analyzeColorDistribution(data, info);

      return layout;
    } catch (error) {
      logger.error("Color layout extraction failed:", error);
      return this.getDefaultColorLayout();
    }
  }

  generateGaussianKernel(sigma) {
    const size = Math.ceil(sigma * 6);
    const kernel = new Float32Array(size);
    const center = Math.floor(size / 2);

    let sum = 0;
    for (let i = 0; i < size; i++) {
      const x = i - center;
      kernel[i] = Math.exp(-(x * x) / (2 * sigma * sigma));
      sum += kernel[i];
    }

    // Normalize kernel
    for (let i = 0; i < size; i++) {
      kernel[i] /= sum;
    }

    return kernel;
  }

  async ensureValidBuffer(data) {
    try {
      // If data is already a buffer, return it
      if (Buffer.isBuffer(data) || data instanceof Uint8Array) {
        return data;
      }

      // Convert to buffer if needed
      return await sharp(data, {
        failOnError: false,
        unlimited: true,
        sequentialRead: true,
      })
        .raw()
        .toBuffer();
    } catch (error) {
      logger.error("Buffer validation failed:", error);
      return new Uint8Array(0);
    }
  }

  applyGaussianBlur(data, width, height) {
    try {
      const sigma = 1.4;
      const size = Math.ceil(sigma * 6);
      const kernel = this.generateGaussianKernel(sigma);
      const result = new Float32Array(data.length);
      const temp = new Float32Array(data.length);

      // Apply horizontal blur
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          let sum = 0;
          let weightSum = 0;

          for (let i = -Math.floor(size / 2); i <= Math.floor(size / 2); i++) {
            const xi = x + i;
            if (xi >= 0 && xi < width) {
              const weight = kernel[i + Math.floor(size / 2)];
              sum += data[y * width + xi] * weight;
              weightSum += weight;
            }
          }

          temp[y * width + x] = weightSum > 0 ? sum / weightSum : 0;
        }
      }

      // Apply vertical blur
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          let sum = 0;
          let weightSum = 0;

          for (let i = -Math.floor(size / 2); i <= Math.floor(size / 2); i++) {
            const yi = y + i;
            if (yi >= 0 && yi < height) {
              const weight = kernel[i + Math.floor(size / 2)];
              sum += temp[yi * width + x] * weight;
              weightSum += weight;
            }
          }

          result[y * width + x] = weightSum > 0 ? sum / weightSum : 0;
        }
      }

      return result;
    } catch (error) {
      logger.error("Gaussian blur failed:", error);
      return new Float32Array(data.length);
    }
  }

  async applyCannyEdgeDetection(data, width) {
    try {
      if (!data || !width) {
        return new Float32Array(0);
      }

      const height = Math.floor(data.length / width);
      if (height < 2 || width < 2) {
        return new Float32Array(width * height);
      }

      // Create a new buffer from data
      const buffer = Buffer.from(data.buffer);
      const processedBuffer = await sharp(buffer, {
        raw: {
          width,
          height,
          channels: 1,
        },
        failOnError: false,
      })
        .raw()
        .toBuffer();

      // Ensure valid buffer for processing
      if (!Buffer.isBuffer(processedBuffer)) {
        return new Float32Array(width * height);
      }

      // Apply Gaussian smoothing
      const smoothed = await this.applyGaussianBlur(
        processedBuffer,
        width,
        height
      );

      // Calculate gradients
      const [magnitude, direction] = this.calculateGradients(smoothed, width);

      // Apply non-maximum suppression
      const suppressed = this.applyNonMaxSuppression(
        magnitude,
        direction,
        width
      );

      // Double thresholding and edge tracking
      return this.performHysteresisThresholding(
        suppressed,
        width,
        this.config.EDGE.CANNY_LOW,
        this.config.EDGE.CANNY_HIGH
      );
    } catch (error) {
      logger.error("Canny edge detection failed:", error);
      return new Float32Array(width * height || 0);
    }
  }

  async extractTextureRegions(features) {
    try {
      const regions = [];
      const visited = new Set();
      const width = features.width || 256;
      const height = features.height || 256;

      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          if (!visited.has(`${x},${y}`)) {
            const region = await this.growTextureRegion(
              features,
              x,
              y,
              visited
            );
            if (this.isSignificantRegion(region)) {
              regions.push(this.analyzeTextureRegion(region));
            }
          }
        }
      }

      return {
        regions,
        statistics: this.calculateRegionStatistics(regions),
        relationships: this.analyzeRegionRelationships(regions),
      };
    } catch (error) {
      console.error("Texture region extraction failed:", error);
      return {
        regions: [],
        statistics: this.getDefaultRegionStatistics(),
        relationships: [],
      };
    }
  }

  calculateClusterConfidence(cluster) {
    if (!cluster || !cluster.points || cluster.points.length === 0) return 0;

    // Calculate average distance to center
    const distances = cluster.points.map((point) =>
      Math.sqrt(
        Math.pow(point[0] - cluster.center[0], 2) +
          Math.pow(point[1] - cluster.center[1], 2) +
          Math.pow(point[2] - cluster.center[2], 2)
      )
    );

    const avgDistance =
      distances.reduce((sum, d) => sum + d, 0) / distances.length;
    const maxDistance = Math.max(...distances);

    // Normalize confidence score (0-1)
    return Math.max(0, Math.min(1, 1 - avgDistance / maxDistance));
  }

  async extractDominantColors(data, info) {
    try {
      const channels = info.channels || 4;
      const pixels = [];

      // Sample pixels (use sampling to improve performance)
      const samplingRate = Math.max(
        1,
        Math.floor(data.length / (channels * 10000))
      );

      for (let i = 0; i < data.length; i += channels * samplingRate) {
        pixels.push({
          r: data[i],
          g: data[i + 1],
          b: data[i + 2],
          a: channels > 3 ? data[i + 3] : 255,
        });
      }

      // Convert to LAB color space for better clustering
      const labColors = pixels.map((pixel) => this.rgbToLAB(pixel));

      // Perform clustering
      const clusters = await this.performMeanShiftClustering(labColors, 8);

      // Convert back to RGB and calculate populations
      return clusters.map((cluster) => ({
        lab: cluster.center,
        rgb: this.labToRGB(cluster.center),
        population:
          cluster.points.length /
          clusters.reduce((sum, c) => sum + c.points.length, 0),
        confidence: this.calculateClusterConfidence(cluster),
      }));
    } catch (error) {
      console.error("Dominant color extraction failed:", error);
      return [];
    }
  }

  analyzeColorDistribution(data, info) {
    try {
      if (!data) {
        throw new Error("No data provided for color distribution analysis");
      }

      const channels = info?.channels || 4;
      const distribution = {
        histogram: Array(channels)
          .fill()
          .map(() => new Float32Array(256).fill(0)),
        entropy: 0,
        uniformity: 0,
      };

      // Safely calculate distribution
      if (data instanceof Uint8Array || data instanceof Buffer) {
        for (let i = 0; i < data.length; i += channels) {
          for (let c = 0; c < channels; c++) {
            if (i + c < data.length) {
              distribution.histogram[c][data[i + c]]++;
            }
          }
        }
      }

      // Normalize histograms
      const pixelCount = Math.floor(data.length / channels);
      distribution.histogram.forEach((hist) => {
        for (let i = 0; i < 256; i++) {
          hist[i] /= pixelCount;
        }
      });

      return distribution;
    } catch (error) {
      console.error("Color distribution analysis failed:", error);
      return {
        histogram: Array(4)
          .fill()
          .map(() => new Float32Array(256).fill(0)),
        entropy: 0,
        uniformity: 0,
      };
    }
  }

  calculateColorPairQuality(color1, color2) {
    if (!color1 || !color2) return 0;

    const contrast = this.calculateColorContrastRatio(color1, color2);
    const harmony = this.calculateColorHarmony(color1, color2);

    return contrast * 0.6 + harmony * 0.4;
  }

  calculateHarmonySetQuality(harmonySet) {
    if (!Array.isArray(harmonySet) || harmonySet.length === 0) return 0;

    const colorPairScores = harmonySet.map((pair) => {
      if (!Array.isArray(pair) || pair.length < 2) return 0;
      return this.calculateColorPairQuality(pair[0], pair[1]);
    });

    return (
      colorPairScores.reduce((sum, score) => sum + score, 0) / harmonySet.length
    );
  }

  calculateHarmonyQuality(harmonies) {
    if (!harmonies) return 0;

    const scores = {
      complementary: this.calculateHarmonySetQuality(
        harmonies.complementary || []
      ),
      analogous: this.calculateHarmonySetQuality(harmonies.analogous || []),
      triadic: this.calculateHarmonySetQuality(harmonies.triadic || []),
    };

    return (
      scores.complementary * 0.4 + scores.analogous * 0.3 + scores.triadic * 0.3
    );
  }

  calculateDistributionQuality(distribution) {
    if (!distribution || !distribution.histograms) return 0;

    const entropy = this.calculateHistogramEntropy(distribution.histograms);
    const uniformity = this.calculateHistogramUniformity(
      distribution.histograms
    );

    return entropy * 0.6 + uniformity * 0.4;
  }

  assessColorQuality(features) {
    if (!features)
      return {
        overall: 0,
        components: { dominant: 0, distribution: 0, harmony: 0 },
      };

    try {
      const dominantQuality = features.dominant
        ? features.dominant.reduce(
            (sum, color) => sum + (color.confidence || 0),
            0
          ) / (features.dominant.length || 1)
        : 0;

      const distributionQuality = features.distribution
        ? this.calculateDistributionQuality(features.distribution)
        : 0;

      const harmonyQuality = features.harmonies
        ? this.calculateHarmonyQuality(features.harmonies)
        : 0;

      const overall =
        dominantQuality * 0.4 +
        distributionQuality * 0.3 +
        harmonyQuality * 0.3;

      return {
        overall: Math.min(1, Math.max(0, overall)),
        components: {
          dominant: dominantQuality,
          distribution: distributionQuality,
          harmony: harmonyQuality,
        },
      };
    } catch (error) {
      logger.error("Color quality assessment failed:", error);
      return {
        overall: 0,
        components: { dominant: 0, distribution: 0, harmony: 0 },
      };
    }
  }

  getDefaultColorFeatures() {
    return {
      statistics: {
        mean: new Float32Array(4).fill(0),
        variance: new Float32Array(4).fill(0),
        skewness: new Float32Array(4).fill(0),
        kurtosis: new Float32Array(4).fill(0),
      },
      dominant: [],
      distribution: {
        histograms: [
          new Float32Array(256).fill(0),
          new Float32Array(256).fill(0),
          new Float32Array(256).fill(0),
          new Float32Array(256).fill(0),
        ],
        entropy: 0,
        uniformity: 0,
      },
      harmonies: {
        complementary: [],
        analogous: [],
        triadic: [],
        splitComplementary: [],
        monochromatic: [],
      },
      quality: {
        overall: 0,
        components: {
          statistics: 0,
          distribution: 0,
          harmony: 0,
        },
        reliability: 0,
      },
    };
  }

  async analyzeColors(data, info) {
    try {
      // Extract and analyze color features
      const colorStatistics = await this.calculateColorStatistics(data, info);
      const dominantColors = await this.findDominantColors(data, info);
      const colorDistribution = await this.analyzeColorDistribution(data, info);
      const colorHarmonies = await this.analyzeColorHarmonies(dominantColors);

      return {
        statistics: colorStatistics,
        dominant: dominantColors,
        distribution: colorDistribution,
        harmonies: colorHarmonies,
        quality: this.assessColorQuality({
          statistics: colorStatistics,
          dominant: dominantColors,
          distribution: colorDistribution,
          harmonies: colorHarmonies,
        }),
      };
    } catch (error) {
      console.error("Color analysis failed:", error);
      return this.getDefaultColorFeatures();
    }
  }

  async extractColorFeatures(buffer) {
    try {
      if (!Buffer.isBuffer(buffer)) {
        throw new Error("Invalid input: Buffer expected");
      }

      // Process in a single Sharp pipeline
      const { data, info } = await sharp(buffer, {
        failOnError: false,
        sequentialRead: true,
      })
        .ensureAlpha()
        .raw()
        .toBuffer({ resolveWithObject: true });

      // Extract features with timeouts
      const [dominantColors, distribution] = await Promise.all([
        this.withTimeout(this.findDominantColors(data, info), 5000),
        this.withTimeout(this.analyzeColorDistribution(data, info), 5000),
      ]);

      const features = {
        dominant: dominantColors,
        distribution: distribution,
        statistics: await this.calculateColorStatistics(data, info),
      };

      return {
        ...features,
        quality: this.assessColorQuality(features),
      };
    } catch (error) {
      logger.error("Color feature extraction failed:", error);
      return this.getDefaultColorFeatures();
    }
  }

  calculateLocalContinuity(edges, x, y, width) {
    const directions = [
      [-1, -1],
      [-1, 0],
      [-1, 1],
      [0, -1],
      [0, 1],
      [1, -1],
      [1, 0],
      [1, 1],
    ];

    let connectedEdges = 0;
    let totalWeight = 0;

    for (const [dx, dy] of directions) {
      const idx = (y + dy) * width + (x + dx);
      if (edges[idx] > 0) {
        const weight = Math.abs(dx) + Math.abs(dy) === 1 ? 1.0 : 0.707;
        connectedEdges += weight;
        totalWeight += weight;
      }
    }

    return totalWeight > 0 ? connectedEdges / totalWeight : 0;
  }

  analyzeEdgeContinuity(edges, width) {
    if (!edges || !edges.length) {
      return {
        score: 0,
        breaks: [],
        continuityMap: new Float32Array(0),
      };
    }

    const continuityMap = new Float32Array(edges.length);
    const breaks = [];
    let totalContinuity = 0;
    let edgePoints = 0;
    const height = Math.floor(edges.length / width);

    // Analyze local continuity
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const idx = y * width + x;
        if (edges[idx] > 0) {
          edgePoints++;
          const continuityScore = this.calculateLocalContinuity(
            edges,
            x,
            y,
            width
          );
          continuityMap[idx] = continuityScore;
          totalContinuity += continuityScore;

          if (continuityScore < 0.5) {
            breaks.push({ x, y, severity: 1 - continuityScore });
          }
        }
      }
    }

    return {
      score: edgePoints > 0 ? totalContinuity / edgePoints : 0,
      breaks,
      continuityMap,
    };
  }

  findConnectedNeighbors(edges, x, y, width) {
    const directions = [
      [-1, -1],
      [-1, 0],
      [-1, 1],
      [0, -1],
      [0, 1],
      [1, -1],
      [1, 0],
      [1, 1],
    ];

    const neighbors = [];
    for (const [dx, dy] of directions) {
      const idx = (y + dy) * width + (x + dx);
      if (edges[idx] > 0) {
        neighbors.push({ x: x + dx, y: y + dy, strength: edges[idx] });
      }
    }

    return neighbors;
  }

  calculateJunctionAngles(neighbors) {
    if (neighbors.length < 2) return [];

    const angles = neighbors.map((n) => {
      return (Math.atan2(n.y, n.x) * 180) / Math.PI;
    });

    // Sort angles
    angles.sort((a, b) => a - b);

    // Calculate angle differences
    const differences = [];
    for (let i = 0; i < angles.length; i++) {
      const next = (i + 1) % angles.length;
      let diff = angles[next] - angles[i];
      if (diff < 0) diff += 360;
      differences.push(diff);
    }

    return differences;
  }

  calculateUniformity(histogram) {
    return histogram.reduce((sum, p) => sum + p * p, 0);
  }

  calculateDistributionStats(values) {
    if (!values || values.length === 0) {
      return {
        mean: 0,
        std: 0,
        entropy: 0,
        uniformity: 0,
      };
    }

    const mean = mathUtils.calculateMean(values);
    const std = Math.sqrt(mathUtils.calculateVariance(values, mean));

    // Calculate histogram for entropy and uniformity
    const histogram = this.calculateHistogram(values, 10);
    const entropy = this.calculateEntropy(histogram);
    const uniformity = this.calculateUniformity(histogram);

    return {
      mean,
      std,
      entropy,
      uniformity,
    };
  }

  analyzeJunctionDistribution(junctions, width, height) {
    if (junctions.length === 0) {
      return {
        spatial: { x: [], y: [] },
        strength: [],
        angles: [],
      };
    }

    // Analyze spatial distribution
    const xCoords = junctions.map((j) => j.x / width);
    const yCoords = junctions.map((j) => j.y / height);
    const strengths = junctions.map((j) => j.strength);
    const allAngles = junctions.flatMap((j) => j.angles);

    return {
      spatial: {
        x: this.calculateDistributionStats(xCoords),
        y: this.calculateDistributionStats(yCoords),
      },
      strength: this.calculateDistributionStats(strengths),
      angles: this.calculateDistributionStats(allAngles),
    };
  }

  detectJunctionPoints(edges, width) {
    if (!edges || !edges.length) {
      return {
        points: [],
        density: 0,
        distribution: {},
      };
    }

    const height = Math.floor(edges.length / width);
    const junctions = [];
    const visited = new Set();

    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const idx = y * width + x;
        if (edges[idx] > 0 && !visited.has(idx)) {
          const neighbors = this.findConnectedNeighbors(edges, x, y, width);
          if (neighbors.length >= 3) {
            junctions.push({
              x,
              y,
              strength: edges[idx],
              connections: neighbors.length,
              angles: this.calculateJunctionAngles(neighbors),
            });
          }
          visited.add(idx);
        }
      }
    }

    return {
      points: junctions,
      density: junctions.length / (width * height),
      distribution: this.analyzeJunctionDistribution(junctions, width, height),
    };
  }

  calculateImageGradients(data, width) {
    const height = Math.floor(data.length / width);
    const Ix = new Float32Array(data.length);
    const Iy = new Float32Array(data.length);

    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const idx = y * width + x;
        Ix[idx] = (data[idx + 1] - data[idx - 1]) / 2;
        Iy[idx] = (data[(y + 1) * width + x] - data[(y - 1) * width + x]) / 2;
      }
    }

    return [Ix, Iy];
  }

  calculateHarrisMatrix(Ix, Iy, x, y, width, windowSize) {
    let M00 = 0,
      M01 = 0,
      M10 = 0,
      M11 = 0;

    for (let wy = -windowSize; wy <= windowSize; wy++) {
      for (let wx = -windowSize; wx <= windowSize; wx++) {
        const idx = (y + wy) * width + (x + wx);
        const ix = Ix[idx];
        const iy = Iy[idx];

        M00 += ix * ix;
        M01 += ix * iy;
        M11 += iy * iy;
      }
    }
    M10 = M01;

    return [M00, M01, M10, M11];
  }

  nonMaximaSuppression(corners, width, height) {
    const radius = 3;
    const suppressed = [];
    const used = new Set();

    // Sort corners by response strength
    corners.sort((a, b) => b.response - a.response);

    for (const corner of corners) {
      const key = `${corner.x},${corner.y}`;
      if (used.has(key)) continue;

      // Check if this is a local maximum
      let isMax = true;
      for (let dy = -radius; dy <= radius; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
          if (dx === 0 && dy === 0) continue;

          const nx = corner.x + dx;
          const ny = corner.y + dy;
          const nkey = `${nx},${ny}`;

          if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
            used.add(nkey);
          }
        }
      }

      if (isMax) {
        suppressed.push(corner);
      }
    }

    return suppressed;
  }

  calculateCornerClustering(corners, width, height) {
    if (corners.length < 2) return 0;

    // Calculate average nearest neighbor distance
    let totalDistance = 0;
    for (const corner of corners) {
      let minDist = Infinity;
      for (const other of corners) {
        if (corner === other) continue;
        const dist = Math.sqrt(
          Math.pow(corner.x - other.x, 2) + Math.pow(corner.y - other.y, 2)
        );
        minDist = Math.min(minDist, dist);
      }
      totalDistance += minDist;
    }

    const avgDistance = totalDistance / corners.length;
    const expectedDistance = Math.sqrt((width * height) / (2 * corners.length));

    return expectedDistance > 0 ? avgDistance / expectedDistance : 0;
  }

  analyzeCornerDistribution(corners, width, height) {
    if (corners.length === 0) {
      return {
        spatial: { x: [], y: [] },
        strength: [],
        clustering: 0,
      };
    }

    // Analyze spatial distribution
    const xCoords = corners.map((c) => c.x / width);
    const yCoords = corners.map((c) => c.y / height);
    const strengths = corners.map((c) => c.response);

    const spatial = {
      x: this.calculateDistributionStats(xCoords),
      y: this.calculateDistributionStats(yCoords),
    };

    const strength = this.calculateDistributionStats(strengths);
    const clustering = this.calculateCornerClustering(corners, width, height);

    return {
      spatial,
      strength,
      clustering,
    };
  }

  detectCornerPoints(edges, width, height) {
    if (!edges || !edges.length) {
      return {
        points: [],
        strength: [],
        density: 0,
        distribution: {},
      };
    }

    const corners = [];
    const windowSize = 3;
    const k = 0.04; // Harris detector free parameter

    // Calculate gradients
    const [Ix, Iy] = this.calculateImageGradients(edges, width);

    for (let y = windowSize; y < height - windowSize; y++) {
      for (let x = windowSize; x < width - windowSize; x++) {
        const [M00, M01, M10, M11] = this.calculateHarrisMatrix(
          Ix,
          Iy,
          x,
          y,
          width,
          windowSize
        );

        // Calculate corner response
        const det = M00 * M11 - M01 * M10;
        const trace = M00 + M11;
        const response = det - k * trace * trace;

        if (response > 0.1) {
          // Threshold for corner detection
          corners.push({ x, y, response });
        }
      }
    }

    // Non-maxima suppression
    const finalCorners = this.nonMaximaSuppression(corners, width, height);

    return {
      points: finalCorners,
      strength: finalCorners.map((c) => c.response),
      density: finalCorners.length / (width * height),
      distribution: this.analyzeCornerDistribution(finalCorners, width, height),
    };
  }

  assessWaveletQuality(coefficients) {
    if (!coefficients || !coefficients.coefficients) {
      return {
        quality: 0,
        components: {
          energyDistribution: { score: 0 },
          stability: 0,
          compression: 0,
        },
      };
    }

    const energyDistribution = this.analyzeWaveletEnergy(
      coefficients.coefficients
    );
    const stabilityScore = this.assessWaveletStability(
      coefficients.coefficients
    );
    const compressionRatio =
      this.calculateWaveletCompressionRatio(coefficients);

    return {
      quality:
        energyDistribution.score * 0.4 +
        stabilityScore * 0.3 +
        compressionRatio * 0.3,
      components: {
        energyDistribution,
        stability: stabilityScore,
        compression: compressionRatio,
      },
    };
  }

  assessHashQuality(hashes) {
    return {
      dct: this.assessDCTHashQuality(hashes.dct),
      wavelet: this.assessWaveletHashQuality(hashes.wavelet),
      radon: this.assessRadonHashQuality(hashes.radon),
      combined: this.assessCombinedHashQuality(hashes),
    };
  }

  async validateAndPrepareBuffer(buffer) {
    if (!buffer) {
      throw new Error("No buffer provided");
    }

    if (!Buffer.isBuffer(buffer)) {
      if (buffer instanceof Uint8Array) {
        buffer = Buffer.from(buffer);
      } else {
        throw new Error("Invalid input: Buffer expected");
      }
    }

    try {
      const validBuffer = await this.prepareValidBuffer(buffer);
      const metadata = await sharp(validBuffer, {
        failOnError: false,
      }).metadata();

      if (!metadata || !metadata.width || !metadata.height) {
        throw new Error("Invalid image metadata");
      }

      return {
        buffer: validBuffer,
        metadata,
      };
    } catch (error) {
      throw new Error(`Buffer validation failed: ${error.message}`);
    }
  }

  assessWaveletStability(coefficients) {
    try {
      if (!coefficients) return 0;
      const energyDistribution = this.analyzeWaveletEnergy(coefficients);
      const entropyScore = this.calculateWaveletEntropy(coefficients);
      return energyDistribution.score * 0.6 + entropyScore * 0.4;
    } catch (error) {
      console.error("Wavelet stability assessment failed:", error);
      return 0;
    }
  }

  async convertAndProcessBuffer(buffer) {
    try {
      // Try to convert to JPEG
      const converted = await sharp(buffer)
        .jpeg(this.optimizationConfig.convertOptions)
        .toBuffer();

      // Get metadata of converted image
      const metadata = await sharp(converted).metadata();

      if (!metadata.width || !metadata.height) {
        throw new Error("Invalid image dimensions after conversion");
      }

      return this.processBuffer(converted, metadata);
    } catch (convertError) {
      // If JPEG fails, try PNG
      try {
        const converted = await sharp(buffer).png().toBuffer();

        const metadata = await sharp(converted).metadata();

        if (!metadata.width || !metadata.height) {
          throw new Error("Invalid image dimensions after PNG conversion");
        }

        return this.processBuffer(converted, metadata);
      } catch (pngError) {
        throw new Error(`Format conversion failed: ${convertError.message}`);
      }
    }
  }

  async processBuffer(buffer, metadata) {
    // Normalize the image
    const processed = await sharp(buffer, {
      failOnError: false, // Be more lenient after conversion
    })
      .ensureAlpha()
      .raw()
      .toBuffer();

    return {
      buffer: processed,
      metadata,
      originalFormat: metadata.format,
    };
  }

  async _processBufferSafely(buffer) {
    try {
      const processed = await this.validateAndPreprocessBuffer(buffer);
      if (!processed || !processed.buffer) {
        throw new Error("Buffer processing failed");
      }
      return processed.buffer;
    } catch (error) {
      logger.error("Safe buffer processing failed:", error);
      throw error;
    }
  }

  async _convertBuffer(buffer) {
    try {
      console.log("Starting buffer conversion...");

      // Add some basic Sharp configuration
      sharp.cache(false);
      sharp.concurrency(1);

      // First try to detect format
      let detectedFormat;
      try {
        const header = buffer.slice(0, 12);
        if (header.indexOf(Buffer.from("PNG")) === 1) detectedFormat = "png";
        else if (header.indexOf(Buffer.from("JFIF")) === 6)
          detectedFormat = "jpeg";
        else if (header.indexOf(Buffer.from("Exif")) === 6)
          detectedFormat = "jpeg";
        else if (header.indexOf(Buffer.from("GIF8")) === 0)
          detectedFormat = "gif";
        else if (header.indexOf(Buffer.from("WEBP")) > 0)
          detectedFormat = "webp";
      } catch (err) {
        console.log("Format detection failed:", err.message);
      }

      // Initial conversion attempt with strict limits
      const initialOptions = {
        failOnError: false,
        limitInputPixels: 512 * 512 * 1024, // 512MP limit
        sequentialRead: true,
        density: 72, // Start with lower density
      };

      // Try raw pixel data recovery first
      try {
        const raw = await sharp(buffer, {
          failOnError: false,
          raw: {
            width: 1,
            height: 1,
            channels: 4,
          },
        })
          .raw()
          .toBuffer({ resolveWithObject: true })
          .catch(() => null);

        if (raw) {
          const { data, info } = raw;
          const converted = await sharp(data, {
            raw: {
              width: info.width || 1,
              height: info.height || 1,
              channels: info.channels || 4,
            },
          })
            .jpeg()
            .toBuffer();

          // Verify conversion
          const metadata = await sharp(converted, {
            failOnError: false,
          }).metadata();
          if (metadata?.width && metadata?.height) {
            console.log("Raw pixel recovery successful");
            return converted;
          }
        }
      } catch (err) {
        console.log(
          "Raw pixel recovery failed, trying format-specific conversion"
        );
      }

      // Try format-specific optimized conversion
      try {
        const image = sharp(buffer, {
          failOnError: false,
          unlimited: true,
          sequentialRead: true,
        });

        if (detectedFormat) {
          image.toFormat(detectedFormat, {
            quality: 100,
            lossless: true,
          });
        }

        const converted = await image.toBuffer();
        const metadata = await sharp(converted).metadata();

        if (metadata?.width && metadata?.height) {
          console.log(
            `Format-specific conversion successful: ${detectedFormat}`
          );
          return converted;
        }
      } catch (err) {
        console.log(
          "Format-specific conversion failed, trying TIFF conversion"
        );
      }

      // Try TIFF conversion
      try {
        const converted = await sharp(buffer, {
          failOnError: false,
          unlimited: true,
        })
          .tiff({
            compression: "lzw",
            predictor: "horizontal",
          })
          .toBuffer();

        const metadata = await sharp(converted).metadata();
        if (metadata?.width && metadata?.height) {
          console.log("TIFF conversion successful");
          return converted;
        }
      } catch (err) {
        console.log("TIFF conversion failed, trying high-quality JPEG");
      }

      // Try high-quality JPEG
      try {
        const converted = await sharp(buffer, {
          failOnError: false,
          unlimited: true,
        })
          .jpeg({
            quality: 100,
            chromaSubsampling: "4:4:4",
            mozjpeg: true,
          })
          .toBuffer();

        const metadata = await sharp(converted).metadata();
        if (metadata?.width && metadata?.height) {
          console.log("JPEG conversion successful");
          return converted;
        }
      } catch (err) {
        console.log("JPEG conversion failed, trying lossless WebP");
      }

      // Try lossless WebP
      try {
        const converted = await sharp(buffer, {
          failOnError: false,
          unlimited: true,
        })
          .webp({
            lossless: true,
            quality: 100,
            effort: 6,
          })
          .toBuffer();

        const metadata = await sharp(converted).metadata();
        if (metadata?.width && metadata?.height) {
          console.log("WebP conversion successful");
          return converted;
        }
      } catch (err) {
        console.log("WebP conversion failed, trying fallback options");
      }

      // Last resort: Fallback with different options
      const fallbackOptions = {
        failOnError: false,
        limitInputPixels: false,
        sequentialRead: true,
        density: 300,
      };

      const converted = await sharp(buffer, fallbackOptions)
        .jpeg({
          quality: 90,
          chromaSubsampling: "4:4:4",
          force: true,
        })
        .toBuffer();

      const metadata = await sharp(converted, {
        failOnError: false,
      }).metadata();
      if (!metadata?.width || !metadata?.height) {
        throw new Error("Conversion verification failed");
      }

      console.log("Fallback conversion successful");
      return converted;
    } catch (error) {
      console.error("Buffer conversion failed:", error);
      throw error;
    }
  }

  _calculateDimensions(width, height, maxSize) {
    const aspect = width / height;

    if (width > maxSize) {
      width = maxSize;
      height = Math.round(width / aspect);
    }

    if (height > maxSize) {
      height = maxSize;
      width = Math.round(height * aspect);
    }

    return { width, height };
  }

  async validateAndPreprocessBuffer(buffer) {
    if (!Buffer.isBuffer(buffer)) {
      throw new Error("Invalid input: Buffer expected");
    }

    try {
      // Try processing with multiple formats
      const processingAttempts = [
        // Try JPEG first
        async () => {
          return await sharp(buffer, { failOnError: false })
            .jpeg({ quality: 90, chromaSubsampling: "4:4:4" })
            .toBuffer();
        },
        // Try PNG
        async () => {
          return await sharp(buffer, { failOnError: false }).png().toBuffer();
        },
        // Try WebP
        async () => {
          return await sharp(buffer, { failOnError: false })
            .webp({ quality: 90 })
            .toBuffer();
        },
      ];

      let processedBuffer;
      let error;

      for (const attempt of processingAttempts) {
        try {
          processedBuffer = await attempt();
          // Verify the processed buffer
          const metadata = await sharp(processedBuffer, {
            failOnError: false,
          }).metadata();
          if (metadata && metadata.width && metadata.height) {
            return {
              buffer: processedBuffer,
              metadata,
            };
          }
        } catch (err) {
          error = err;
          continue;
        }
      }

      // If all attempts fail, try one last time with raw processing
      return await sharp(buffer, {
        failOnError: false,
        unlimited: true,
        sequentialRead: true,
        density: 300,
      })
        .raw()
        .toBuffer({ resolveWithObject: true });
    } catch (error) {
      throw new Error(`Buffer preprocessing failed: ${error.message}`);
    }
  }

  // 2. Add missing functions for region analysis

  calculateCellColorVariance(colors) {
    // Ensure colors is an array
    if (!Array.isArray(colors)) {
      return 0;
    }

    // Filter valid colors
    const validColors = colors.filter(
      (color) =>
        color &&
        typeof color === "object" &&
        typeof color.r === "number" &&
        typeof color.g === "number" &&
        typeof color.b === "number"
    );

    if (validColors.length === 0) {
      return 0;
    }

    const meanColor = this.calculateMeanColor(validColors);
    let variance = 0;

    for (const color of validColors) {
      variance += Math.pow(color.r - meanColor.r, 2);
      variance += Math.pow(color.g - meanColor.g, 2);
      variance += Math.pow(color.b - meanColor.b, 2);
    }

    return variance / (validColors.length * 3);
  }

  calculateMeanColor(colors) {
    const sum = colors.reduce(
      (acc, color) => ({
        r: acc.r + color.r,
        g: acc.g + color.g,
        b: acc.b + color.b,
      }),
      { r: 0, g: 0, b: 0 }
    );

    return {
      r: sum.r / colors.length,
      g: sum.g / colors.length,
      b: sum.b / colors.length,
    };
  }

  // 3. Add quality assessment functions

  assessDominantColorQuality(dominantColors) {
    if (!dominantColors || !Array.isArray(dominantColors)) return 0;

    const coverage = dominantColors.reduce(
      (sum, color) => sum + (color.population || 0),
      0
    );
    const confidence =
      dominantColors.reduce((sum, color) => sum + (color.confidence || 0), 0) /
      Math.max(1, dominantColors.length);

    return coverage * 0.6 + confidence * 0.4;
  }

  normalizeDistribution(distribution) {
    if (!distribution) return new Map();

    try {
      if (distribution instanceof Map) {
        return distribution;
      }

      if (Array.isArray(distribution)) {
        const map = new Map();
        distribution.forEach((value, index) => {
          map.set(index, value);
        });
        return map;
      }

      if (typeof distribution === "object") {
        return new Map(Object.entries(distribution));
      }

      return new Map();
    } catch (error) {
      console.error("Error normalizing distribution:", error);
      return new Map();
    }
  }

  calculateDistributionUniformity(distribution) {
    const normalizedDist = this.normalizeDistribution(distribution);
    if (!normalizedDist || normalizedDist.size === 0) return 0;

    const values = Array.from(normalizedDist.values());
    const sum = values.reduce((a, b) => a + b, 0);

    if (sum === 0) return 0;

    // Calculate normalized entropy
    const probabilities = values.map((v) => v / sum);
    const entropy = -probabilities.reduce(
      (e, p) => e + (p > 0 ? p * Math.log2(p) : 0),
      0
    );

    // Normalize entropy to [0,1]
    const maxEntropy = Math.log2(normalizedDist.size);
    return maxEntropy > 0 ? entropy / maxEntropy : 0;
  }

  assessDistributionQuality(distribution) {
    if (!distribution) return 0;

    const entropy = this.calculateDistributionEntropy(distribution);
    const uniformity = this.calculateDistributionUniformity(distribution);

    return entropy * 0.6 + uniformity * 0.4;
  }

  assessHarmonyQuality(harmonies) {
    if (!harmonies) return 0;

    const complementaryScore = this.assessHarmonySet(harmonies.complementary);
    const analogousScore = this.assessHarmonySet(harmonies.analogous);
    const triadicScore = this.assessHarmonySet(harmonies.triadic);

    return complementaryScore * 0.4 + analogousScore * 0.3 + triadicScore * 0.3;
  }

  // 4. Helper functions for quality assessment
  calculateDistributionEntropy(distribution) {
    if (!distribution || !distribution.histogram) return 0;

    let entropy = 0;
    for (const value of distribution.histogram) {
      if (value > 0) {
        entropy -= value * Math.log2(value);
      }
    }
    return Math.min(1, entropy / 8); // Normalize to [0,1]
  }

  // Add missing Sharp operation wrapper
  async processWithSharp(buffer, operation) {
    if (!buffer) {
      throw new Error("Invalid input: No buffer provided");
    }

    let processedBuffer;
    try {
      // Ensure we have a valid buffer
      if (!Buffer.isBuffer(buffer)) {
        if (buffer instanceof Uint8Array) {
          processedBuffer = Buffer.from(buffer);
        } else {
          throw new Error("Invalid input: Buffer expected");
        }
      } else {
        processedBuffer = buffer;
      }

      // Validate buffer content
      if (processedBuffer.length === 0) {
        throw new Error("Invalid input: Empty buffer");
      }

      // Create Sharp instance with validated buffer
      const sharpInstance = sharp(processedBuffer, {
        failOnError: false,
        unlimited: true,
        sequentialRead: true,
        density: 300,
      });

      // Apply operation with error handling
      const result = await operation(sharpInstance);

      // Verify result if it's a buffer
      if (Buffer.isBuffer(result)) {
        const verifyMetadata = await sharp(result, {
          failOnError: false,
        }).metadata();

        if (
          !verifyMetadata ||
          !verifyMetadata.width ||
          !verifyMetadata.height
        ) {
          throw new Error("Invalid result from Sharp operation");
        }
      }

      return result;
    } catch (error) {
      logger.error("Sharp operation failed:", {
        error: error.message,
        stack: error.stack,
        inputBufferSize: processedBuffer ? processedBuffer.length : 0,
      });
      throw error;
    } finally {
      if (processedBuffer && processedBuffer !== buffer) {
        processedBuffer = null;
      }
    }
  }

  getDefaultCellColorFeatures() {
    return {
      dominant: this.getDefaultDominantColor(),
      distribution: new Map(),
      contrast: 0,
      coherence: 0,
    };
  }

  // Helper methods and utilities
  getDefaultColorLayout() {
    return {
      width: 0,
      height: 0,
      grid: Array(8)
        .fill()
        .map(() =>
          Array(8).fill({
            dominantColor: { r: 0, g: 0, b: 0 },
            colorVariance: 0,
            edgeStrength: 0,
          })
        ),
      dominantColors: [],
      colorDistribution: new Map(),
    };
  }

  assessHarmonySet(harmonies) {
    if (!harmonies || !Array.isArray(harmonies)) return 0;

    return (
      harmonies.reduce((score, harmony) => {
        const contrast = this.calculateColorContrast(harmony[0], harmony[1]);
        const balance = this.calculateColorBalance(harmony[0], harmony[1]);
        return score + (contrast * 0.6 + balance * 0.4);
      }, 0) / Math.max(1, harmonies.length)
    );
  }

  calculateColorContrast(color1, color2) {
    if (!color1 || !color2) return 0;

    const luminance1 = this.calculateLuminance(color1);
    const luminance2 = this.calculateLuminance(color2);

    const brightest = Math.max(luminance1, luminance2);
    const darkest = Math.min(luminance1, luminance2);

    return (brightest + 0.05) / (darkest + 0.05);
  }

  calculateLuminance(color) {
    const r = color.r / 255;
    const g = color.g / 255;
    const b = color.b / 255;

    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
  }

  calculateColorBalance(color1, color2) {
    if (!color1 || !color2) return 0;

    const diff = Math.abs(
      this.calculateLuminance(color1) - this.calculateLuminance(color2)
    );
    return 1 - diff;
  }

  // 5. Update signature generation to use validated input
  calculateWaveletEnergy(coefficients) {
    try {
      if (!coefficients || typeof coefficients !== "object") {
        return {
          distribution: new Map(),
          score: 0,
          concentration: 0,
        };
      }

      const values = [];
      for (const band of Object.values(coefficients)) {
        if (Array.isArray(band) || ArrayBuffer.isView(band)) {
          values.push(...Array.from(band));
        }
      }

      if (values.length === 0) {
        return {
          distribution: new Map(),
          score: 0,
          concentration: 0,
        };
      }

      const energy =
        values.reduce((sum, val) => sum + val * val, 0) / values.length;
      return {
        distribution: new Map([["total", energy]]),
        score: Math.min(1, energy / 255),
        concentration: this.calculateEnergyConcentration(values),
      };
    } catch (error) {
      console.error("Wavelet energy calculation failed:", error);
      return {
        distribution: new Map(),
        score: 0,
        concentration: 0,
      };
    }
  }

  calculateEdgeStrength(edges) {
    if (!edges || !Array.isArray(edges)) {
      const edgesArray = edges ? Array.from(edges) : [];
      const nonZeroEdges = edgesArray.filter((e) => e > 0);
      return nonZeroEdges.length > 0
        ? nonZeroEdges.reduce((sum, val) => sum + val, 0) / nonZeroEdges.length
        : 0;
    }
    const nonZeroEdges = edges.filter((e) => e > 0);
    return nonZeroEdges.length > 0
      ? nonZeroEdges.reduce((sum, val) => sum + val, 0) / nonZeroEdges.length
      : 0;
  }

  analyzeEdgeDistribution(edges, width, height) {
    try {
      if (!edges || !edges.length) {
        return {
          density: 0,
          uniformity: 0,
          distribution: new Float32Array(0),
        };
      }

      // Calculate density map
      const blockSize = 16; // Size of blocks for distribution analysis
      const numBlocksX = Math.ceil(width / blockSize);
      const numBlocksY = Math.ceil(height / blockSize);
      const distribution = new Float32Array(numBlocksX * numBlocksY);

      // Calculate edge density in each block
      for (let y = 0; y < numBlocksY; y++) {
        for (let x = 0; x < numBlocksX; x++) {
          let blockSum = 0;
          let count = 0;

          for (let by = 0; by < blockSize; by++) {
            for (let bx = 0; bx < blockSize; bx++) {
              const px = x * blockSize + bx;
              const py = y * blockSize + by;
              if (px < width && py < height) {
                const idx = py * width + px;
                if (edges[idx] > 0) {
                  blockSum += edges[idx];
                  count++;
                }
              }
            }
          }

          distribution[y * numBlocksX + x] = count > 0 ? blockSum / count : 0;
        }
      }

      // Calculate overall density
      const density = edges.filter((e) => e > 0).length / edges.length;

      // Calculate uniformity using distribution variance
      const mean =
        distribution.reduce((a, b) => a + b, 0) / distribution.length;
      const variance =
        distribution.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) /
        distribution.length;
      const uniformity = Math.exp(-variance);

      return {
        density,
        uniformity,
        distribution,
      };
    } catch (error) {
      console.error("Edge distribution analysis failed:", error);
      return {
        density: 0,
        uniformity: 0,
        distribution: new Float32Array(0),
      };
    }
  }

  applyCannyEdgeDetection(data, width) {
    const height = Math.floor(data.length / width);

    // Apply Gaussian blur for noise reduction
    const smoothed = this.applyGaussianBlur(data, width, height);

    // Calculate gradients
    const [magnitude, direction] = this.calculateGradients(smoothed, width);

    // Apply non-maximum suppression
    const suppressed = this.applyNonMaxSuppression(magnitude, direction, width);

    // Perform double thresholding and edge tracking
    return this.performHysteresisThresholding(
      suppressed,
      width,
      this.config.EDGE.CANNY_LOW,
      this.config.EDGE.CANNY_HIGH
    );
  }

  // Helper method for edge detection
  async _detectEdges(data, width, height) {
    try {
      // Apply Gaussian blur
      const smoothed = await this.applyGaussianBlur(data, width, height);

      // Calculate gradients
      const [magnitude, direction] = this.calculateGradients(smoothed, width);

      // Apply non-maximum suppression
      const suppressed = this.applyNonMaxSuppression(
        magnitude,
        direction,
        width
      );

      // Double thresholding and edge tracking
      return this.performHysteresisThresholding(
        suppressed,
        width,
        this.config.EDGE.CANNY_LOW,
        this.config.EDGE.CANNY_HIGH
      );
    } catch (error) {
      console.error("Edge detection processing failed:", error);
      return new Float32Array(data.length).fill(0);
    }
  }

  extractCellTexture(textureRegions, x, y, width, height) {
    try {
      // Ensure textureRegions is an array
      const regions = Array.isArray(textureRegions) ? textureRegions : [];

      const cellFeatures = {
        glcm: [],
        lbp: [],
        gabor: [],
        statistics: {},
      };

      const cellRect = { x, y, width, height };

      // Process each valid region
      for (const region of regions) {
        if (this.regionOverlapsCell(region, cellRect)) {
          if (region.glcm) cellFeatures.glcm.push(region.glcm);
          if (region.lbp) cellFeatures.lbp.push(region.lbp);
          if (region.gabor) cellFeatures.gabor.push(region.gabor);
        }
      }

      return cellFeatures;
    } catch (error) {
      console.error("Cell texture extraction failed:", error);
      return {
        glcm: [],
        lbp: [],
        gabor: [],
        statistics: {},
      };
    }
  }

  analyzeTextureRegularity(cellTexture) {
    try {
      if (!cellTexture) return 0;

      const regularity = {
        glcm: this.calculateGLCMRegularity(cellTexture.glcm || []),
        lbp: this.calculateLBPRegularity(cellTexture.lbp || []),
        gabor: this.calculateGaborRegularity(cellTexture.gabor || []),
      };

      return (
        regularity.glcm * 0.4 + regularity.lbp * 0.3 + regularity.gabor * 0.3
      );
    } catch (error) {
      console.error("Texture regularity analysis failed:", error);
      return 0;
    }
  }

  calculateGLCMRegularity(glcmFeatures) {
    if (!Array.isArray(glcmFeatures) || glcmFeatures.length === 0) return 0;

    const homogeneities = glcmFeatures.map((f) => f.homogeneity || 0);
    const mean =
      homogeneities.reduce((a, b) => a + b, 0) / homogeneities.length;
    const variance =
      homogeneities.reduce((sum, h) => sum + Math.pow(h - mean, 2), 0) /
      homogeneities.length;

    return Math.exp(-variance * 2); // Higher regularity = lower variance
  }

  calculateLBPRegularity(lbpFeatures) {
    if (!Array.isArray(lbpFeatures) || lbpFeatures.length === 0) return 0;

    const patterns = lbpFeatures.map((f) => f.histogram || []).flat();
    if (patterns.length === 0) return 0;

    const sum = patterns.reduce((a, b) => a + b, 0);
    if (sum === 0) return 0;

    // Calculate normalized entropy as measure of regularity
    const entropy = patterns.reduce((e, count) => {
      const p = count / sum;
      return e - (p > 0 ? p * Math.log2(p) : 0);
    }, 0);

    return 1 - entropy / Math.log2(patterns.length);
  }

  calculateGaborRegularity(gaborFeatures) {
    if (!Array.isArray(gaborFeatures) || gaborFeatures.length === 0) return 0;

    const responses = gaborFeatures.map((f) => f.magnitude || 0);
    const mean = responses.reduce((a, b) => a + b, 0) / responses.length;
    const variance =
      responses.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) /
      responses.length;

    return 1 / (1 + Math.sqrt(variance) / mean);
  }

  calculateEnergy(values) {
    if (!values || values.length === 0) return 0;
    return values.reduce((sum, val) => sum + val * val, 0) / values.length;
  }

  calculateHistogramStatistics(histogram) {
    if (!histogram || !histogram.length) {
      return {
        mean: 0,
        variance: 0,
        entropy: 0,
      };
    }

    const sum = histogram.reduce((a, b) => a + b, 0);
    if (sum === 0) {
      return {
        mean: 0,
        variance: 0,
        entropy: 0,
      };
    }

    // Normalize histogram
    const normalized = histogram.map((v) => v / sum);

    // Calculate statistics
    const mean = this.calculateMean(normalized);
    const variance = this.calculateVariance(normalized);
    const entropy = -normalized.reduce(
      (e, p) => e + (p > 0 ? p * Math.log2(p) : 0),
      0
    );

    return {
      mean,
      variance,
      entropy,
    };
  }

  calculateTextureStatistics(features) {
    const stats = {
      mean: 0,
      variance: 0,
      entropy: 0,
      energy: 0,
    };

    try {
      // Calculate GLCM statistics
      if (features.glcm && features.glcm.features) {
        stats.mean += features.glcm.features.mean || 0;
        stats.energy += features.glcm.features.energy || 0;
        stats.entropy += features.glcm.features.entropy || 0;
      }

      // Calculate LBP statistics
      if (features.lbp && features.lbp.histogram) {
        const lbpStats = this.calculateHistogramStatistics(
          features.lbp.histogram
        );
        stats.mean += lbpStats.mean;
        stats.variance += lbpStats.variance;
        stats.entropy += lbpStats.entropy;
      }

      // Calculate Gabor statistics
      if (features.gabor && features.gabor.features) {
        const magnitudes = features.gabor.features
          .map((f) => f.magnitude)
          .filter((m) => typeof m === "number");

        if (magnitudes.length > 0) {
          stats.mean += this.calculateMean(magnitudes);
          stats.variance += this.calculateVariance(magnitudes);
          stats.energy += this.calculateEnergy(magnitudes);
        }
      }

      // Normalize combined statistics
      const numFeatures = Object.values(features).filter(Boolean).length;
      if (numFeatures > 0) {
        stats.mean /= numFeatures;
        stats.variance /= numFeatures;
        stats.entropy /= numFeatures;
        stats.energy /= numFeatures;
      }

      return stats;
    } catch (error) {
      console.error("Error calculating texture statistics:", error);
      return stats;
    }
  }

  calculateAverageFeature(features, featureName) {
    if (!Array.isArray(features) || features.length === 0) return 0;
    const values = features.map((f) => f[featureName] || 0);
    return values.reduce((a, b) => a + b, 0) / values.length;
  }

  regionOverlapsCell(region, cell) {
    if (!region || !region.bounds) return false;

    return !(
      region.bounds.maxX < cell.x ||
      region.bounds.minX > cell.x + cell.width ||
      region.bounds.maxY < cell.y ||
      region.bounds.minY > cell.y + cell.height
    );
  }

  async validateImageFormat(buffer, originalFormat) {
    const supportedFormats = new Set([
      "jpeg",
      "jpg",
      "png",
      "webp",
      "tiff",
      "gif",
      "svg",
      "heic",
      "bmp",
    ]);

    try {
      // Basic validation
      if (!Buffer.isBuffer(buffer)) {
        throw new Error("Invalid input: Buffer expected");
      }

      if (buffer.length === 0) {
        throw new Error("Invalid input: Empty buffer");
      }

      if (buffer.length > 100 * 1024 * 1024) {
        // 100MB limit
        throw new Error("Buffer exceeds size limit of 100MB");
      }

      // Format detection and validation
      const formatChecks = {
        jpeg: async (buf) => {
          // Check JPEG markers (SOI and EOI)
          return (
            buf[0] === 0xff &&
            buf[1] === 0xd8 &&
            buf[buf.length - 2] === 0xff &&
            buf[buf.length - 1] === 0xd9
          );
        },
        jpg: async (buf) => {
          return await formatChecks.jpeg(buf);
        },
        png: async (buf) => {
          // Check PNG signature
          const signature = buf.slice(0, 8).toString("hex");
          return signature === "89504e470d0a1a0a";
        },
        webp: async (buf) => {
          // Check WEBP signature
          const signature = buf.slice(0, 4).toString("ascii");
          return (
            signature === "RIFF" &&
            buf.slice(8, 12).toString("ascii") === "WEBP"
          );
        },
        gif: async (buf) => {
          // Check GIF signature
          const signature = buf.slice(0, 6).toString("ascii");
          return signature.startsWith("GIF8");
        },
        tiff: async (buf) => {
          // Check TIFF signature
          const signature = buf.slice(0, 4).toString("hex");
          return signature === "49492a00" || signature === "4d4d002a";
        },
        heic: async (buf) => {
          // Check HEIC signature (ftyp box with heic brand)
          if (buf.length < 12) return false;
          const brand = buf.slice(8, 12).toString("ascii");
          return brand === "heic" || brand === "heix" || brand === "hevc";
        },
        bmp: async (buf) => {
          // Check BMP signature
          return buf[0] === 0x42 && buf[1] === 0x4d;
        },
      };

      // Try to get metadata using Sharp
      const sharp = require("sharp");
      const metadata = await sharp(buffer, {
        failOnError: false,
        unlimited: true,
        sequentialRead: true,
      }).metadata();

      if (!metadata || !metadata.format) {
        throw new Error("Unable to detect image format");
      }

      const detectedFormat = metadata.format.toLowerCase();

      // Validate detected format against supported formats
      if (!supportedFormats.has(detectedFormat)) {
        throw new Error(`Unsupported format: ${detectedFormat}`);
      }

      // Perform format-specific validation if available
      if (formatChecks[detectedFormat]) {
        const isValidFormat = await formatChecks[detectedFormat](buffer);
        if (!isValidFormat) {
          throw new Error(
            `Invalid ${detectedFormat.toUpperCase()} file structure`
          );
        }
      }

      // Validate image dimensions
      if (!metadata.width || !metadata.height) {
        throw new Error("Invalid image dimensions");
      }

      if (metadata.width < 8 || metadata.height < 8) {
        throw new Error("Image dimensions too small (minimum 8x8)");
      }

      if (metadata.width > 20000 || metadata.height > 20000) {
        throw new Error("Image dimensions too large (maximum 20000x20000)");
      }

      // Validate pixel count
      const pixelCount = metadata.width * metadata.height;
      if (pixelCount > 40000000) {
        // 40MP limit
        throw new Error("Image pixel count exceeds maximum allowed (40MP)");
      }

      // Validate color depth and channels
      if (
        metadata.channels &&
        (metadata.channels < 1 || metadata.channels > 4)
      ) {
        throw new Error("Invalid number of color channels");
      }

      if (metadata.depth && (metadata.depth < 1 || metadata.depth > 16)) {
        throw new Error("Invalid color depth");
      }

      // Check for color profile if available
      if (
        metadata.space &&
        !["srgb", "rgb", "cmyk", "gray"].includes(metadata.space.toLowerCase())
      ) {
        throw new Error("Unsupported color space");
      }

      // Additional integrity check
      try {
        await sharp(buffer, { failOnError: true })
          .resize(10, 10) // Small resize to verify image can be processed
          .toBuffer();
      } catch (integrityError) {
        throw new Error(
          `Image integrity check failed: ${integrityError.message}`
        );
      }

      return {
        valid: true,
        format: detectedFormat,
        metadata: {
          width: metadata.width,
          height: metadata.height,
          channels: metadata.channels,
          depth: metadata.depth,
          space: metadata.space,
          hasAlpha: metadata.hasAlpha,
          orientation: metadata.orientation,
        },
        buffer: buffer,
      };
    } catch (error) {
      logger.error("Format validation failed:", {
        error: error.message,
        stack: error.stack,
        originalFormat,
        detectedFormat: metadata?.format,
      });

      // Don't throw, return validation result
      return {
        valid: false,
        error: error.message,
        originalFormat,
        detectedFormat: metadata?.format,
      };
    }
  }

  async checkImageIntegrity(buffer, metadata) {
    try {
      // Try to decode a small portion of the image
      await sharp(buffer, { failOnError: true }).resize(10, 10).toBuffer();

      // Check for corrupt EXIF data
      if (metadata.exif) {
        try {
          await sharp(buffer).metadata();
        } catch (exifError) {
          return {
            valid: false,
            reason: "Corrupt EXIF data detected",
          };
        }
      }

      return { valid: true };
    } catch (error) {
      return {
        valid: false,
        reason: `Image decoding failed: ${error.message}`,
      };
    }
  }

  validateColorSpace(space) {
    const recommendedSpaces = new Set(["srgb", "rgb"]);
    const valid = recommendedSpaces.has(space.toLowerCase());

    return {
      valid,
      recommended: valid ? space : "sRGB",
    };
  }

  estimateMemoryRequirement(metadata) {
    const bytesPerPixel =
      ((metadata.channels || 4) * (metadata.depth || 8)) / 8;
    const baseMemory = metadata.width * metadata.height * bytesPerPixel;

    // Account for processing overhead
    const processingMultiplier = 3; // Buffer copies and processing overhead
    const estimatedTotal = baseMemory * processingMultiplier;

    // Add safety margin
    return estimatedTotal * 1.2;
  }

  async validateInput(buffer) {
    const startTime = Date.now();
    let metadata = null;

    try {
      logger.info("Starting input validation...");

      // 1. Basic buffer validation
      if (!Buffer.isBuffer(buffer)) {
        throw new ValidationError("Invalid input: Buffer expected");
      }

      if (buffer.length === 0) {
        throw new ValidationError("Invalid input: Empty buffer");
      }

      if (buffer.length > 100 * 1024 * 1024) {
        // 100MB limit
        throw new ValidationError(
          "Buffer size exceeds maximum allowed (100MB)"
        );
      }

      // 2. Format detection and validation
      try {
        metadata = await sharp(buffer, {
          failOnError: false,
          limitInputPixels: this.processingOptions.limitInputPixels,
          sequentialRead: true,
        }).metadata();

        if (!metadata || !metadata.width || !metadata.height) {
          throw new ValidationError("Invalid image metadata");
        }

        // Log detected format info
        logger.info("Detected image format:", {
          format: metadata.format,
          width: metadata.width,
          height: metadata.height,
          channels: metadata.channels,
          space: metadata.space,
          depth: metadata.depth,
          density: metadata.density,
          hasAlpha: metadata.hasAlpha,
          orientation: metadata.orientation,
        });
      } catch (metadataError) {
        throw new ValidationError(
          `Failed to extract metadata: ${metadataError.message}`
        );
      }

      // 3. Dimension validation
      if (
        metadata.width < this.validationThresholds.minDimension ||
        metadata.height < this.validationThresholds.minDimension
      ) {
        throw new ValidationError(
          `Image too small: Minimum dimensions are ${this.validationThresholds.minDimension}x${this.validationThresholds.minDimension}`
        );
      }

      if (
        metadata.width > this.validationThresholds.maxDimension ||
        metadata.height > this.validationThresholds.maxDimension
      ) {
        throw new ValidationError(
          `Image too large: Maximum dimensions are ${this.validationThresholds.maxDimension}x${this.validationThresholds.maxDimension}`
        );
      }

      // 4. Pixel count validation
      const pixelCount = metadata.width * metadata.height;
      if (pixelCount > this.validationThresholds.maxPixels) {
        throw new ValidationError(
          `Image pixel count (${pixelCount}) exceeds maximum allowed (${this.validationThresholds.maxPixels})`
        );
      }

      // 5. Format-specific validation
      const formatValidation = await this.validateImageFormat(
        metadata.format,
        buffer
      );
      if (!formatValidation.valid) {
        throw new ValidationError(
          `Format validation failed: ${formatValidation.reason}`
        );
      }

      // 6. Image integrity check
      const integrityCheck = await this.checkImageIntegrity(buffer, metadata);
      if (!integrityCheck.valid) {
        throw new ValidationError(
          `Image integrity check failed: ${integrityCheck.reason}`
        );
      }

      // 7. Color space validation
      if (metadata.space) {
        const colorSpaceValidation = this.validateColorSpace(metadata.space);
        if (!colorSpaceValidation.valid) {
          logger.warn("Non-optimal color space detected:", {
            detected: metadata.space,
            recommended: colorSpaceValidation.recommended,
          });
        }
      }

      // 8. Memory requirement estimation
      const memoryEstimate = this.estimateMemoryRequirement(metadata);
      const availableMemory =
        process.memoryUsage().heapTotal - process.memoryUsage().heapUsed;

      if (memoryEstimate > availableMemory * 0.8) {
        throw new ValidationError(
          `Insufficient memory: Required ${Math.round(
            memoryEstimate / 1024 / 1024
          )}MB, ` + `Available ${Math.round(availableMemory / 1024 / 1024)}MB`
        );
      }

      logger.info("Input validation successful:", {
        duration: Date.now() - startTime + "ms",
        format: metadata.format,
        dimensions: `${metadata.width}x${metadata.height}`,
        memoryEstimate: Math.round(memoryEstimate / 1024 / 1024) + "MB",
      });

      return {
        buffer,
        metadata,
        memoryEstimate,
        validationDuration: Date.now() - startTime,
      };
    } catch (error) {
      logger.error("Input validation failed:", {
        error: error.message,
        stack: error.stack,
        duration: Date.now() - startTime + "ms",
      });

      if (error instanceof ValidationError) {
        throw error;
      }
      throw new ValidationError(`Image validation failed: ${error.message}`);
    }
  }

  getDefaultHashFeatures() {
    const hashSize = Math.ceil(this.config.HASH.HASH_SIZE / 8);
    return {
      dct: {
        hash: new Uint8Array(hashSize),
        quality: { robustness: 0, distinctiveness: 0, stability: 0 },
      },
      wavelet: {
        hash: new Uint8Array(hashSize),
        quality: { robustness: 0, distinctiveness: 0, stability: 0 },
      },
      radon: {
        hash: new Uint8Array(hashSize),
        quality: { robustness: 0, distinctiveness: 0, stability: 0 },
      },
      combined: {
        hash: new Uint8Array(hashSize),
        quality: { robustness: 0, distinctiveness: 0, stability: 0 },
      },
    };
  }

  // Enhanced signature generation with proper error handling

  getDefaultSignature() {
    return {
      version: "2.0",
      timestamp: Date.now(),
      metadata: {},
      colors: this.getDefaultColorFeatures(),
      edges: this.getDefaultEdgeFeatures(),
      textures: this.getDefaultTextureFeatures(),
      hashes: this.getDefaultHashFeatures(),
      spatial: this.getDefaultSpatialFeatures(),
      quality: {
        overall: 0,
        components: {
          color: 0,
          edge: 0,
          texture: 0,
          hash: 0,
          spatial: 0,
        },
      },
    };
  }

  async cleanupBuffer(buffer) {
    try {
      if (Buffer.isBuffer(buffer)) {
        buffer.fill(0);
      }
    } catch (error) {
      logger.warn("Buffer cleanup failed:", error);
    }
  }

  async generatePerceptualHashes(buffer) {
    try {
      if (!Buffer.isBuffer(buffer)) {
        throw new Error("Invalid input: Buffer expected");
      }

      const processedBuffer = await this.processWithSharp(
        buffer,
        async (image) => {
          return await image
            .resize(this.config.HASH.DCT_SIZE, this.config.HASH.DCT_SIZE, {
              fit: "fill",
              kernel: "lanczos3",
            })
            .grayscale()
            .raw()
            .toBuffer();
        }
      );

      // Generate different types of hashes in parallel
      const [dctHash, waveletHash, radonHash] = await Promise.all([
        this.generateDCTHash(processedBuffer),
        this.generateWaveletHash(processedBuffer),
        this.generateRadonHash(processedBuffer),
      ]);

      // Generate combined hash
      const combinedHash = await this.generateCombinedHash({
        dct: dctHash,
        wavelet: waveletHash,
        radon: radonHash,
      });

      return {
        dct: dctHash,
        wavelet: waveletHash,
        radon: radonHash,
        combined: combinedHash,
      };
    } catch (error) {
      logger.error("Hash generation failed:", {
        error: error.message,
        stack: error.stack,
      });
      return this.getDefaultHashFeatures();
    }
  }

  calculateComponentReliability(qualities) {
    const reliabilities = Object.values(qualities)
      .map((q) => (q && typeof q === "object" ? q.reliability || 0 : 0))
      .filter((r) => !isNaN(r));

    return reliabilities.length > 0
      ? reliabilities.reduce((sum, r) => sum + r, 0) / reliabilities.length
      : 0;
  }

  calculateSignatureConfidence(qualities) {
    const confidences = Object.values(qualities)
      .map((q) => (q && typeof q === "object" ? q.confidence || 0 : 0))
      .filter((c) => !isNaN(c));

    return confidences.length > 0
      ? confidences.reduce((sum, c) => sum + c, 0) / confidences.length
      : 0;
  }

  getDefaultQuality() {
    return {
      overall: 0,
      components: {
        color: 0,
        edge: 0,
        texture: 0,
        hash: 0,
        spatial: 0,
      },
      reliability: 0,
      confidence: 0,
    };
  }

  calculateSignatureQuality(signature) {
    if (!signature) {
      return this.getDefaultQuality();
    }

    try {
      const qualities = {
        color: this.assessColorQuality(signature.colors),
        edge: this.assessEdgeQuality(signature.edges),
        texture: this.assessTextureQuality(signature.textures),
        hash: this.assessHashQuality(signature.hashes),
        spatial: this.assessSpatialQuality(signature.spatial),
      };

      // Ensure all components have valid values
      Object.keys(qualities).forEach((key) => {
        if (!qualities[key] || typeof qualities[key].overall !== "number") {
          qualities[key] = {
            overall: 0,
            components: {},
            reliability: 0,
          };
        }
      });

      // Calculate combined quality
      const weights = {
        color: 0.25,
        edge: 0.25,
        texture: 0.2,
        hash: 0.15,
        spatial: 0.15,
      };

      const overall = Object.entries(qualities).reduce(
        (sum, [key, quality]) => sum + quality.overall * (weights[key] || 0),
        0
      );

      // Calculate reliability and confidence
      const reliability = this.calculateComponentReliability(qualities);
      const confidence = this.calculateSignatureConfidence(qualities);

      return {
        overall: Math.max(0, Math.min(1, overall)),
        components: qualities,
        reliability: Math.max(0, Math.min(1, reliability)),
        confidence: Math.max(0, Math.min(1, confidence)),
      };
    } catch (error) {
      logger.error("Error calculating signature quality:", error);
      return this.getDefaultQuality();
    }
  }

  async generateSignature(buffer) {
    const startTime = Date.now();
    let processedBuffer = null;

    try {
      if (!buffer) {
        throw new Error("Invalid input: No buffer provided");
      }

      // Validate and preprocess the buffer
      const { buffer: validBuffer, metadata } =
        await this.validateAndPreprocessBuffer(buffer);
      processedBuffer = validBuffer;

      // Extract features in parallel with proper error handling
      const [colorFeatures, edgeFeatures, textureFeatures, hashFeatures] =
        await Promise.all([
          this._safeExtract(() => this.extractColorFeatures(processedBuffer)),
          this._safeExtract(() => this.extractEdgeFeatures(processedBuffer)),
          this._safeExtract(() => this.extractTextureFeatures(processedBuffer)),
          this._safeExtract(() =>
            this.generatePerceptualHashes(processedBuffer)
          ),
        ]);

      // Generate spatial verification
      const spatialFeatures = await this._safeExtract(() =>
        this.generateSpatialVerification(processedBuffer)
      );

      // Construct signature with proper error handling
      const signature = {
        version: "2.0",
        timestamp: Date.now(),
        metadata: {
          width: metadata.width,
          height: metadata.height,
          format: metadata.format,
          size: processedBuffer.length,
          processingTime: Date.now() - startTime,
        },
        colors: colorFeatures || this.getDefaultColorFeatures(),
        edges: edgeFeatures || this.getDefaultEdgeFeatures(),
        textures: textureFeatures || this.getDefaultTextureFeatures(),
        hashes: hashFeatures || this.getDefaultHashFeatures(),
        spatial: spatialFeatures || this.getDefaultSpatialFeatures(),
      };

      // Calculate quality metrics
      signature.quality = await this._safeExtract(() =>
        this.calculateSignatureQuality(signature)
      );

      return signature;
    } catch (error) {
      logger.error("Error in signature generation:", {
        error: error.message,
        stack: error.stack,
        duration: Date.now() - startTime,
      });

      return {
        version: "2.0",
        timestamp: Date.now(),
        metadata: {},
        colors: this.getDefaultColorFeatures(),
        edges: this.getDefaultEdgeFeatures(),
        textures: this.getDefaultTextureFeatures(),
        hashes: this.getDefaultHashFeatures(),
        spatial: this.getDefaultSpatialFeatures(),
        quality: this.getDefaultQuality(),
        error: error.message,
      };
    } finally {
      // Cleanup
      if (processedBuffer && processedBuffer !== buffer) {
        processedBuffer = null;
      }
      if (global.gc) {
        global.gc();
      }
    }
  }

  handleExtractionResult(result, featureType) {
    if (result.status === "fulfilled") {
      return result.value;
    }

    logger.error(`${featureType} feature extraction failed:`, {
      error: result.reason?.message,
      stack: result.reason?.stack,
    });

    switch (featureType) {
      case "color":
        return this.getDefaultColorFeatures();
      case "edge":
        return this.getDefaultEdgeFeatures();
      case "texture":
        return this.getDefaultTextureFeatures();
      default:
        return {};
    }
  }

  validateFeatures(features) {
    const missingFeatures = [];

    if (!features.colors || Object.keys(features.colors).length === 0) {
      missingFeatures.push("colors");
    }
    if (!features.edges || Object.keys(features.edges).length === 0) {
      missingFeatures.push("edges");
    }
    if (!features.textures || Object.keys(features.textures).length === 0) {
      missingFeatures.push("textures");
    }

    return missingFeatures;
  }

  async extractFeatures(buffer) {
    const startTime = Date.now();
    let processedBuffer = null;

    try {
      // Use shared buffer validation
      processedBuffer = await BufferUtils.validateAndConvert(buffer);

      // Extract features with proper error handling
      const [colorFeatures, edgeFeatures, textureFeatures] =
        await Promise.allSettled([
          this._safeExtract(() => this.extractColorFeatures(processedBuffer)),
          this._safeExtract(() => this.extractEdgeFeatures(processedBuffer)),
          this._safeExtract(() => this.extractTextureFeatures(processedBuffer)),
        ]);

      return {
        colors: this.handleExtractionResult(colorFeatures, "color"),
        edges: this.handleExtractionResult(edgeFeatures, "edge"),
        textures: this.handleExtractionResult(textureFeatures, "texture"),
      };
    } catch (error) {
      logger.error("Feature extraction failed:", {
        error: error.message,
        stack: error.stack,
        duration: Date.now() - startTime + "ms",
      });

      return {
        colors: this.getDefaultColorFeatures(),
        edges: this.getDefaultEdgeFeatures(),
        textures: this.getDefaultTextureFeatures(),
      };
    } finally {
      if (processedBuffer) {
        processedBuffer = null;
      }
    }
  }

  async _processFeature(type, buffer) {
    if (!Buffer.isBuffer(buffer)) {
      console.error(`Invalid buffer input to ${type} feature processing`);
      return null;
    }

    try {
      // Process the buffer with sharp
      const { data, info } = await sharp(buffer, {
        failOnError: false,
        unlimited: true,
        sequentialRead: true,
      })
        .ensureAlpha()
        .raw()
        .toBuffer({ resolveWithObject: true });

      switch (type) {
        case "color":
          return await this.extractColorFeatures(data, info);
        case "edge":
          return await this.extractEdgeFeatures(data, info);
        case "texture":
          return await this.extractTextureFeatures(data, info);
        default:
          throw new Error(`Unknown feature type: ${type}`);
      }
    } catch (error) {
      console.error(`${type} feature extraction failed:`, error);
      return null;
    }
  }

  async cleanupTemporaryResources() {
    try {
      // Clear any caches
      this.colorAnalyzer?.cache?.clear();
      this.edgeAnalyzer?.cache?.clear();
      this.textureAnalyzer?.cache?.clear();

      // Force garbage collection if available
      if (global.gc) {
        global.gc();
      }
    } catch (error) {
      console.warn("Resource cleanup warning:", error);
    }
  }

  validateSignatureCompleteness(signature) {
    const requiredFields = [
      "metadata",
      "colors",
      "edges",
      "textures",
      "hashes",
      "spatial",
      "quality",
    ];

    return requiredFields.every((field) => {
      const hasField = Boolean(signature[field]);
      if (!hasField) {
        console.warn(`Missing required field: ${field}`);
      }
      return hasField;
    });
  }

  // Helper method for fallback signature generation
  generateFallbackSignature(error) {
    return {
      metadata: {
        timestamp: Date.now(),
        error: {
          message: error.message,
          stack: error.stack,
          cause: error.cause,
        },
        version: "2.0",
        progress: {
          color: false,
          edge: false,
          texture: false,
          hash: false,
          spatial: false,
        },
      },
      colors: this.getDefaultColorFeatures(),
      edges: this.getDefaultEdgeFeatures(),
      textures: this.getDefaultTextureFeatures(),
      hashes: this.getDefaultHashFeatures(),
      spatial: this.getDefaultSpatialFeatures(),
      quality: {
        overall: 0,
        components: {
          color: 0,
          edge: 0,
          texture: 0,
          hash: 0,
          spatial: 0,
        },
      },
    };
  }

  // Helper method for error logging
  logSignatureError(error, fallbackSignature) {
    const errorContext = {
      timestamp: Date.now(),
      error: {
        name: error.name,
        message: error.message,
        stack: error.stack,
        cause: error.cause,
      },
      signature: {
        metadata: fallbackSignature.metadata,
        quality: fallbackSignature.quality,
      },
    };

    console.error(
      "Signature generation error context:",
      JSON.stringify(errorContext, null, 2)
    );
  }

  // Helper method for checksum calculation
  calculateSignatureChecksum(signature) {
    const crypto = require("crypto");
    const hash = crypto.createHash("sha256");

    // Only include stable fields in checksum
    const forHashing = {
      colors: signature.colors,
      edges: signature.edges,
      textures: signature.textures,
      hashes: signature.hashes,
      spatial: signature.spatial,
    };

    hash.update(JSON.stringify(forHashing));
    return hash.digest("hex");
  }

  async _generateSignatureWithOptimization(buffer) {
    try {
      // Process with standard options after optimization
      const image = sharp(buffer, { failOnError: false });
      const metadata = await image.metadata();

      console.log(
        `Processing image: ${metadata.width}x${metadata.height} ${metadata.format}`
      );

      // Get base image data
      const processedBuffer = await image.ensureAlpha().raw().toBuffer();

      return await this._generateSignatureInternal(processedBuffer, metadata);
    } catch (error) {
      console.error("Error in signature generation:", error);
      throw error;
    }
  }

  // Helper method for signature generation
  async _generateSignatureInternal(buffer, metadata) {
    // Extract features with parallel processing
    const [colorFeatures, edgeFeatures, textureFeatures, perceptualHashes] =
      await Promise.all([
        this._safeExtract(() => this.extractColorFeatures(buffer, metadata)),
        this._safeExtract(() => this.extractEdgeFeatures(buffer, metadata)),
        this._safeExtract(() => this.extractTextureFeatures(buffer, metadata)),
        this._safeExtract(() =>
          this.generatePerceptualHashes(buffer, metadata)
        ),
      ]);

    return {
      metadata,
      colors: colorFeatures,
      edges: edgeFeatures,
      textures: textureFeatures,
      hashes: perceptualHashes,
      quality: this.calculateSignatureQuality({
        colors: colorFeatures,
        edges: edgeFeatures,
        textures: textureFeatures,
        hashes: perceptualHashes,
      }),
    };
  }

  getDefaultFeatures() {
    return {
      colors: this.getDefaultColorFeatures(),
      edges: this.getDefaultEdgeFeatures(),
      textures: this.getDefaultTextureFeatures(),
      hashes: this.getDefaultHashFeatures(),
      metadata: {},
    };
  }

  // 6. Add safe feature extraction wrapper

  // Add missing quality assessment functions

  calculateOverallEdgeQuality(edges) {
    const strengthScore = this.assessEdgeStrength(edges);
    const continuityScore = this.assessEdgeContinuity(edges);
    const distributionScore = this.assessEdgeDistribution(edges);
    const junctionScore = this.assessJunctionQuality(edges);

    return (
      strengthScore * 0.3 +
      continuityScore * 0.3 +
      distributionScore * 0.2 +
      junctionScore * 0.2
    );
  }

  assessEdgeStrength(edges) {
    if (!edges.strength) return 0;

    const { mean, std } = edges.strength;
    const normalized = mean / (std + 1e-6);
    return Math.min(1, normalized / 5);
  }

  assessEdgeContinuity(edges) {
    if (!edges.continuity) return 0;

    const { score, breaks } = edges.continuity;
    const breakDensity = breaks.length / Math.max(1, edges.points?.length || 1);
    return Math.max(0, Math.min(1, score * (1 - breakDensity)));
  }

  assessEdgeDistribution(edges) {
    if (!edges.orientation?.histogram) return 0;

    const histogram = edges.orientation.histogram;
    return this.calculateDistributionUniformity(histogram);
  }

  assessJunctionQuality(edges) {
    if (!edges || !edges.junctions) return 0;

    const density = edges.junctions.density || 0;
    const normalizedDensity = Math.min(1, density * 100);
    const distributionScore = this.calculateDistributionUniformity(
      edges.junctions.distribution || []
    );

    return normalizedDensity * 0.6 + distributionScore * 0.4;
  }

  // Add robust color calculation for arrays
  calculateMeanColor(colors) {
    if (!Array.isArray(colors) || colors.length === 0) {
      return { r: 0, g: 0, b: 0 };
    }

    const validColors = colors.filter(
      (color) =>
        color &&
        typeof color.r === "number" &&
        typeof color.g === "number" &&
        typeof color.b === "number"
    );

    if (validColors.length === 0) {
      return { r: 0, g: 0, b: 0 };
    }

    const sum = validColors.reduce(
      (acc, color) => ({
        r: acc.r + (color.r || 0),
        g: acc.g + (color.g || 0),
        b: acc.b + (color.b || 0),
      }),
      { r: 0, g: 0, b: 0 }
    );

    return {
      r: Math.round(sum.r / validColors.length),
      g: Math.round(sum.g / validColors.length),
      b: Math.round(sum.b / validColors.length),
    };
  }

  // Add helper functions

  // Add missing LBP functions
  calculateLBPVariance(histogram) {
    if (!histogram || histogram.length === 0) return 0;

    const mean =
      histogram.reduce((sum, val) => sum + val, 0) / histogram.length;
    return (
      histogram.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) /
      histogram.length
    );
  }

  // Add missing texture region analysis

  async growTextureRegion(features, startX, startY, visited) {
    const region = {
      points: [],
      textures: [],
      bounds: {
        minX: startX,
        maxX: startX,
        minY: startY,
        maxY: startY,
      },
    };

    const stack = [[startX, startY]];
    const baseTexture = await this.getTextureAt(features, startX, startY);
    const threshold = this.calculateTextureThreshold(baseTexture);

    while (stack.length > 0) {
      const [x, y] = stack.pop();
      const key = `${x},${y}`;

      if (visited.has(key)) continue;
      visited.add(key);

      const currentTexture = await this.getTextureAt(features, x, y);
      if (this.isTextureSimilar(baseTexture, currentTexture, threshold)) {
        region.points.push([x, y]);
        region.textures.push(currentTexture);
        this.updateRegionBounds(region.bounds, x, y);
        this.addValidNeighbors(stack, x, y, features, visited);
      }
    }

    return region;
  }

  // Add missing cell analysis functions
  async extractCellDominantColor(colorLayout, x, y, width, height) {
    try {
      const colors = [];
      const cellWidth = Math.floor(width / 8);
      const cellHeight = Math.floor(height / 8);

      for (let dy = 0; dy < cellHeight; dy++) {
        for (let dx = 0; dx < cellWidth; dx++) {
          const px = x * cellWidth + dx;
          const py = y * cellHeight + dy;
          if (px < width && py < height) {
            colors.push(this.getColorAt(colorLayout, px, py));
          }
        }
      }

      return this.findDominantColor(colors);
    } catch (error) {
      console.error("Cell dominant color extraction failed:", error);
      return {
        color: { r: 0, g: 0, b: 0 },
        confidence: 0,
      };
    }
  }

  findDominantColor(colors) {
    if (!colors || colors.length === 0) {
      return { color: { r: 0, g: 0, b: 0 }, confidence: 0 };
    }

    // Group similar colors
    const groups = new Map();
    for (const color of colors) {
      let foundGroup = false;
      for (const [key, group] of groups.entries()) {
        if (this.isColorSimilar(color, JSON.parse(key), 30)) {
          group.colors.push(color);
          group.count++;
          foundGroup = true;
          break;
        }
      }
      if (!foundGroup) {
        groups.set(JSON.stringify(color), {
          colors: [color],
          count: 1,
        });
      }
    }

    // Find group with highest count
    let maxCount = 0;
    let dominantGroup = null;
    for (const [key, group] of groups.entries()) {
      if (group.count > maxCount) {
        maxCount = group.count;
        dominantGroup = {
          color: JSON.parse(key),
          colors: group.colors,
        };
      }
    }

    if (!dominantGroup) {
      return { color: colors[0], confidence: 1 };
    }

    // Calculate average color of dominant group
    const avgColor = this.calculateAverageColor(dominantGroup.colors);
    const confidence = maxCount / colors.length;

    return {
      color: avgColor,
      confidence,
    };
  }

  // Add helper functions
  calculateAverageColor(colors) {
    const sum = colors.reduce(
      (acc, color) => ({
        r: acc.r + color.r,
        g: acc.g + color.g,
        b: acc.b + color.b,
      }),
      { r: 0, g: 0, b: 0 }
    );

    return {
      r: Math.round(sum.r / colors.length),
      g: Math.round(sum.g / colors.length),
      b: Math.round(sum.b / colors.length),
    };
  }

  isColorSimilar(color1, color2, threshold = 30) {
    return (
      Math.abs(color1.r - color2.r) <= threshold &&
      Math.abs(color1.g - color2.g) <= threshold &&
      Math.abs(color1.b - color2.b) <= threshold
    );
  }

  async getTextureAt(features, x, y) {
    // Implement texture feature extraction at specific point
    try {
      const idx = y * features.width + x;
      return {
        contrast: features.glcm?.features?.[idx]?.contrast || 0,
        homogeneity: features.glcm?.features?.[idx]?.homogeneity || 0,
        energy: features.glcm?.features?.[idx]?.energy || 0,
        correlation: features.glcm?.features?.[idx]?.correlation || 0,
      };
    } catch (error) {
      console.error("Error getting texture at point:", error);
      return {
        contrast: 0,
        homogeneity: 0,
        energy: 0,
        correlation: 0,
      };
    }
  }

  isTextureSimilar(texture1, texture2, threshold = 0.3) {
    const diff =
      Math.abs(texture1.contrast - texture2.contrast) +
      Math.abs(texture1.homogeneity - texture2.homogeneity) +
      Math.abs(texture1.energy - texture2.energy) +
      Math.abs(texture1.correlation - texture2.correlation);
    return diff <= threshold;
  }

  // Update the main signature generation to use these new functions

  getBit(hash, x, y, size) {
    const bitIndex = y * size + x;
    const byteIndex = Math.floor(bitIndex / 8);
    const bitOffset = 7 - (bitIndex % 8);
    return (hash[byteIndex] >> bitOffset) & 1;
  }

  setBit(hash, x, y, size, value) {
    const bitIndex = y * size + x;
    const byteIndex = Math.floor(bitIndex / 8);
    const bitOffset = 7 - (bitIndex % 8);

    if (value) {
      hash[byteIndex] |= 1 << bitOffset;
    } else {
      hash[byteIndex] &= ~(1 << bitOffset);
    }
  }

  simulateRotation(hash, angle) {
    const size = Math.floor(Math.sqrt(hash.length * 8));
    const rotated = new Uint8Array(hash.length);
    const centerX = size / 2;
    const centerY = size / 2;
    const rad = (angle * Math.PI) / 180;
    const cos = Math.cos(rad);
    const sin = Math.sin(rad);

    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        // Rotate around center
        const dx = x - centerX;
        const dy = y - centerY;
        const newX = Math.round(dx * cos - dy * sin + centerX);
        const newY = Math.round(dx * sin + dy * cos + centerY);

        if (newX >= 0 && newX < size && newY >= 0 && newY < size) {
          const srcBit = this.getBit(hash, x, y, size);
          this.setBit(rotated, newX, newY, size, srcBit);
        }
      }
    }
    return rotated;
  }

  calculateLBPEntropy(histogram) {
    let entropy = 0;
    const sum = histogram.reduce((a, b) => a + b, 0);

    if (sum === 0) return 0;

    for (const count of histogram) {
      if (count > 0) {
        const p = count / sum;
        entropy -= p * Math.log2(p);
      }
    }
    return entropy;
  }

  calculateGradientDistinctiveness(gradients) {
    if (!gradients || !gradients.length) return 0;

    // Convert to array if needed
    const values = Array.isArray(gradients) ? gradients : Array.from(gradients);

    // Calculate mean and variance
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const variance =
      values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) /
      values.length;

    // Calculate normalized distinctiveness score
    return Math.min(1, Math.sqrt(variance) / mean);
  }

  // Optimize the wavelet transform for better performance
  async computeWaveletTransform(data, width, height) {
    const coefficients = {
      LL: new Float32Array(Math.ceil((width * height) / 4)),
      LH: new Float32Array(Math.ceil((width * height) / 4)),
      HL: new Float32Array(Math.ceil((width * height) / 4)),
      HH: new Float32Array(Math.ceil((width * height) / 4)),
    };

    // Process in smaller chunks to prevent hanging
    const chunkSize = 1024;
    for (let y = 0; y < height; y += 2) {
      for (let x = 0; x < width; x += chunkSize) {
        const endX = Math.min(x + chunkSize, width);
        await this.processWaveletChunk(data, coefficients, x, y, endX, width);
      }
    }

    return coefficients;
  }

  async processWaveletChunk(data, coefficients, startX, y, endX, width) {
    const outputWidth = Math.floor(width / 2);

    for (let x = startX; x < endX; x += 2) {
      const i = y * width + x;
      const outputIdx = (y / 2) * outputWidth + x / 2;

      if (outputIdx >= coefficients.LL.length) continue;

      // Calculate wavelet coefficients
      const a = data[i] || 0;
      const b = data[i + 1] || 0;
      const c = data[i + width] || 0;
      const d = data[i + width + 1] || 0;

      coefficients.LL[outputIdx] = (a + b + c + d) / 4;
      coefficients.LH[outputIdx] = (a + b - c - d) / 4;
      coefficients.HL[outputIdx] = (a - b + c - d) / 4;
      coefficients.HH[outputIdx] = (a - b - c + d) / 4;
    }
  }

  calculateGridStatistics(grid) {
    if (!grid || !grid.length)
      return {
        coverage: 0,
        uniformity: 0,
        complexity: 0,
      };

    const cellValues = [];
    let totalCoverage = 0;
    let totalComplexity = 0;

    grid.forEach((row) => {
      row.forEach((cell) => {
        if (cell && cell.features) {
          const importance = cell.features.importance || 0;
          cellValues.push(importance);
          totalCoverage += importance > 0 ? 1 : 0;
          totalComplexity += this.calculateCellComplexity(cell);
        }
      });
    });

    const coverage = totalCoverage / (grid.length * grid[0].length);
    const uniformity = this.calculateGridUniformity(cellValues);
    const complexity = totalComplexity / (grid.length * grid[0].length);

    return {
      coverage,
      uniformity,
      complexity,
      distribution: this.analyzeGridDistribution(cellValues),
    };
  }

  calculateCellComplexity(cell) {
    if (!cell || !cell.features) return 0;

    const colorComplexity = cell.color
      ? this.calculateColorComplexity(cell.color)
      : 0;
    const edgeComplexity = cell.edges
      ? this.calculateEdgeComplexity(cell.edges)
      : 0;
    const textureComplexity = cell.texture
      ? this.calculateTextureComplexity(cell.texture)
      : 0;

    return (
      colorComplexity * 0.4 + edgeComplexity * 0.3 + textureComplexity * 0.3
    );
  }

  calculateSkewness(values) {
    if (!Array.isArray(values) || values.length === 0) return 0;
    const mean = this.calculateMean(values);
    const std = Math.sqrt(this.calculateVariance(values, mean));
    if (std === 0) return 0;

    const sumCubes = values.reduce(
      (acc, val) => acc + Math.pow(val - mean, 3),
      0
    );
    return sumCubes / values.length / Math.pow(std, 3);
  }

  analyzeGridDistribution(values) {
    if (!values || !values.length)
      return {
        entropy: 0,
        variance: 0,
        skewness: 0,
      };

    const mean = mathUtils.calculateMean(values);
    const variance = mathUtils.calculateVariance(values, mean);
    const std = Math.sqrt(variance);

    return {
      entropy: this.calculateEntropy(values),
      variance,
      skewness: mathUtils.calculateSkewness(values, mean, std),
    };
  }

  analyzeGridRelationships(grid) {
    if (!grid || !Array.isArray(grid) || grid.length === 0) {
      return {
        neighbors: [],
        transitions: [],
        patterns: [],
        statistics: {
          meanSimilarity: 0,
          consistencyScore: 0,
        },
      };
    }

    const relationships = [];
    const gridSize = grid.length;

    // Analyze relationships between adjacent cells
    for (let y = 0; y < gridSize; y++) {
      for (let x = 0; x < gridSize; x++) {
        const currentCell = grid[y][x];

        // Check right neighbor
        if (x < gridSize - 1) {
          relationships.push(
            this.analyzeCellPairRelationship(
              currentCell,
              grid[y][x + 1],
              "horizontal"
            )
          );
        }

        // Check bottom neighbor
        if (y < gridSize - 1) {
          relationships.push(
            this.analyzeCellPairRelationship(
              currentCell,
              grid[y + 1][x],
              "vertical"
            )
          );
        }
      }
    }

    // Calculate statistics
    const similarities = relationships
      .map((r) => r.similarity)
      .filter((s) => !isNaN(s));
    const meanSimilarity =
      similarities.length > 0
        ? similarities.reduce((a, b) => a + b, 0) / similarities.length
        : 0;

    const consistencyScore = this.calculateGridConsistency(relationships);

    return {
      relationships,
      statistics: {
        meanSimilarity,
        consistencyScore,
      },
      patterns: this.detectGridPatterns(grid),
    };
  }

  analyzeCellPairRelationship(cell1, cell2, direction) {
    if (!cell1 || !cell2) {
      return {
        similarity: 0,
        transition: "none",
        strength: 0,
        direction,
      };
    }

    // Calculate feature similarities
    const colorSimilarity = this.calculateColorSimilarity(
      cell1.color?.dominant,
      cell2.color?.dominant
    );

    const edgeSimilarity = this.calculateEdgeSimilarity(
      cell1.edges,
      cell2.edges
    );

    const textureSimilarity = this.calculateTextureSimilarity(
      cell1.texture,
      cell2.texture
    );

    // Weight and combine similarities
    const similarity =
      colorSimilarity * 0.4 + edgeSimilarity * 0.3 + textureSimilarity * 0.3;

    // Analyze transition characteristics
    const transition = this.characterizeTransition(cell1, cell2);
    const strength = this.calculateTransitionStrength(cell1, cell2);

    return {
      similarity,
      transition,
      strength,
      direction,
      features: {
        color: colorSimilarity,
        edge: edgeSimilarity,
        texture: textureSimilarity,
      },
    };
  }

  calculateColorSimilarity(color1, color2) {
    if (!color1 || !color2) return 0;

    // Use LAB color space for perceptual color difference
    const deltaE = mathUtils.deltaE(
      color1.lab || { l: 0, a: 0, b: 0 },
      color2.lab || { l: 0, a: 0, b: 0 }
    );

    // Convert deltaE to similarity score (0-1)
    return Math.max(0, 1 - deltaE / 100);
  }

  calculateEdgeSimilarity(edges1, edges2) {
    if (!edges1 || !edges2) return 0;

    // Compare edge characteristics
    const densitySimilarity = 1 - Math.abs(edges1.density - edges2.density);
    const strengthSimilarity = 1 - Math.abs(edges1.strength - edges2.strength);
    const orientationSimilarity = this.compareOrientations(
      edges1.orientation,
      edges2.orientation
    );

    return (
      densitySimilarity * 0.3 +
      strengthSimilarity * 0.3 +
      orientationSimilarity * 0.4
    );
  }

  calculateTextureSimilarity(texture1, texture2) {
    if (!texture1 || !texture2) return 0;

    // Compare texture patterns
    const patternSimilarity = this.compareTexturePatterns(
      texture1.patterns,
      texture2.patterns
    );

    // Compare texture complexity
    const complexitySimilarity =
      1 - Math.abs(texture1.complexity - texture2.complexity);

    // Compare regularity
    const regularitySimilarity =
      1 - Math.abs(texture1.regularity - texture2.regularity);

    return (
      patternSimilarity * 0.4 +
      complexitySimilarity * 0.3 +
      regularitySimilarity * 0.3
    );
  }

  characterizeTransition(cell1, cell2) {
    const similarity = this.analyzeCellSimilarity(cell1, cell2);

    if (similarity > 0.8) return "continuous";
    if (similarity > 0.5) return "gradual";
    if (similarity > 0.2) return "moderate";
    return "sharp";
  }

  calculateTransitionStrength(cell1, cell2) {
    const similarity = this.analyzeCellSimilarity(cell1, cell2);
    return 1 - similarity; // Higher strength for more distinct transitions
  }

  analyzeCellSimilarity(cell1, cell2) {
    if (!cell1 || !cell2) return 0;

    const colorSim = this.calculateColorSimilarity(
      cell1.color?.dominant,
      cell2.color?.dominant
    );

    const edgeSim = this.calculateEdgeSimilarity(cell1.edges, cell2.edges);

    const textureSim = this.calculateTextureSimilarity(
      cell1.texture,
      cell2.texture
    );

    return colorSim * 0.4 + edgeSim * 0.3 + textureSim * 0.3;
  }

  calculateGridConsistency(relationships) {
    if (!relationships || relationships.length === 0) return 0;

    const similarities = relationships.map((r) => r.similarity);
    const mean = similarities.reduce((a, b) => a + b, 0) / similarities.length;

    // Calculate variance of similarities
    const variance =
      similarities.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) /
      similarities.length;

    // Convert variance to consistency score (higher variance = lower consistency)
    return Math.exp(-variance * 5); // Scale factor of 5 gives good spread
  }

  analyzeColorTransitions(colors) {
    const transitions = [];
    for (let i = 0; i < colors.length - 1; i++) {
      transitions.push({
        distance: mathUtils.deltaE(colors[i], colors[i + 1]),
        direction: this.getColorTransitionDirection(colors[i], colors[i + 1]),
        magnitude: this.getColorTransitionMagnitude(colors[i], colors[i + 1]),
      });
    }
    return transitions;
  }

  compareTransitions(t1, t2) {
    const distanceSimilarity = 1 - Math.abs(t1.distance - t2.distance) / 100;
    const directionSimilarity = this.compareDirections(
      t1.direction,
      t2.direction
    );
    const magnitudeSimilarity = 1 - Math.abs(t1.magnitude - t2.magnitude);

    return (distanceSimilarity + directionSimilarity + magnitudeSimilarity) / 3;
  }

  compareTransitionSegments(segment1, segment2) {
    if (segment1.length !== segment2.length) return 0;

    const similarities = segment1.map((t1, i) => {
      const t2 = segment2[i];
      return this.compareTransitions(t1, t2);
    });

    return mathUtils.calculateMean(similarities);
  }

  evaluatePatternPeriod(transitions, period) {
    const segments = [];
    for (let i = 0; i < transitions.length - period + 1; i += period) {
      segments.push(transitions.slice(i, i + period));
    }

    if (segments.length < 2) {
      return { significance: 0, consistency: 0 };
    }

    // Compare all segments with the first one
    const consistencyScores = segments
      .slice(1)
      .map((segment) => this.compareTransitionSegments(segments[0], segment));

    const consistency = mathUtils.calculateMean(consistencyScores);
    const significance = consistency * (segments.length / transitions.length);

    return { significance, consistency };
  }

  detectRepeatingPattern(transitions) {
    // Check for repeating patterns of different lengths
    const maxPeriod = Math.floor(transitions.length / 2);
    let bestPeriod = 1;
    let maxSignificance = 0;
    let bestConsistency = 0;

    for (let period = 1; period <= maxPeriod; period++) {
      const { significance, consistency } = this.evaluatePatternPeriod(
        transitions,
        period
      );
      if (significance > maxSignificance) {
        maxSignificance = significance;
        bestPeriod = period;
        bestConsistency = consistency;
      }
    }

    return {
      significance: maxSignificance,
      period: bestPeriod,
      consistency: bestConsistency,
    };
  }

  calculateDirectionConsistency(directions) {
    if (directions.length < 2) return 0;

    const diffs = directions
      .slice(1)
      .map((dir, i) => Math.abs(dir - directions[i]));

    const meanDiff = mathUtils.calculateMean(diffs);
    return Math.exp(-meanDiff);
  }

  calculateMagnitudeSmoothing(magnitudes) {
    if (magnitudes.length < 2) return 0;

    const diffs = magnitudes
      .slice(1)
      .map((mag, i) => Math.abs(mag - magnitudes[i]));

    const meanDiff = mathUtils.calculateMean(diffs);
    return 1 / (1 + meanDiff);
  }

  detectGradientPattern(transitions) {
    // Check if transitions form a consistent gradient
    const directions = transitions.map((t) => t.direction);
    const magnitudes = transitions.map((t) => t.magnitude);

    const directionConsistency = this.calculateDirectionConsistency(directions);
    const magnitudeSmoothing = this.calculateMagnitudeSmoothing(magnitudes);

    const significance = (directionConsistency + magnitudeSmoothing) / 2;

    return {
      significance,
      direction: this.calculateAverageDirection(directions),
      smoothness: magnitudeSmoothing,
    };
  }

  analyzeColorPattern(cells) {
    const colorValues = cells
      .map((cell) => cell.color?.dominant)
      .filter((color) => color);

    if (colorValues.length < 2) {
      return { type: "none", significance: 0 };
    }

    // Analyze color transitions
    const transitions = this.analyzeColorTransitions(colorValues);

    // Detect patterns
    const repeating = this.detectRepeatingPattern(transitions);
    const gradient = this.detectGradientPattern(transitions);

    if (repeating.significance > gradient.significance) {
      return {
        type: "repeating",
        significance: repeating.significance,
        period: repeating.period,
        consistency: repeating.consistency,
      };
    } else {
      return {
        type: "gradient",
        significance: gradient.significance,
        direction: gradient.direction,
        smoothness: gradient.smoothness,
      };
    }
  }

  calculateOrientationConsistency(orientations) {
    if (!orientations || orientations.length === 0) return 0;

    const mean = this.calculateMean(orientations);
    const deviations = orientations.map((o) => Math.abs(o - mean));
    const avgDeviation = this.calculateMean(deviations);

    return Math.exp(-avgDeviation / 90); // Normalize to [0,1]
  }

  detectGradualOrientationChange(orientations) {
    if (
      !orientations ||
      !Array.isArray(orientations) ||
      orientations.length < 2
    ) {
      return {
        significance: 0,
        consistency: 0,
        direction: 0,
      };
    }

    try {
      // Calculate angular differences between consecutive orientations
      const differences = [];
      for (let i = 1; i < orientations.length; i++) {
        const diff = Math.abs(
          orientations[i].angle - orientations[i - 1].angle
        );
        differences.push(Math.min(diff, 360 - diff));
      }

      // Check for consistent change pattern
      const meanDiff = this.calculateMean(differences);
      const variance = this.calculateVariance(differences);
      const consistency = Math.exp(-variance / (meanDiff + 1e-6));

      // Determine direction of change
      const direction = differences.reduce((acc, diff, i) => {
        if (i === 0) return 0;
        return acc + Math.sign(diff - differences[i - 1]);
      }, 0);

      // Calculate significance based on consistency and magnitude
      const significance = consistency * Math.min(1, meanDiff / 90);

      return {
        significance,
        consistency,
        direction: Math.sign(direction),
      };
    } catch (error) {
      console.error("Error detecting gradual orientation change:", error);
      return {
        significance: 0,
        consistency: 0,
        direction: 0,
      };
    }
  }

  analyzeEdgeOrientationPattern(edges) {
    const orientations = edges.map((edge) => edge.orientation?.dominant || []);

    // Check for consistent orientation
    const consistency = this.calculateOrientationConsistency(orientations);

    // Check for gradual orientation change
    const gradualChange = this.detectGradualOrientationChange(orientations);

    return {
      type:
        gradualChange.significance > consistency.significance
          ? "gradual"
          : "consistent",
      significance: Math.max(
        gradualChange.significance,
        consistency.significance
      ),
      features: {
        consistency,
        gradualChange,
      },
    };
  }

  analyzeEdgePattern(cells) {
    const edgeFeatures = cells.map((cell) => cell.edges).filter((edge) => edge);

    if (edgeFeatures.length < 2) {
      return { type: "none", significance: 0 };
    }

    // Analyze edge continuity
    const continuity = this.analyzeEdgeContinuity(edgeFeatures);

    // Analyze edge orientation patterns
    const orientations = this.analyzeEdgeOrientationPattern(edgeFeatures);

    return {
      type:
        orientations.significance > continuity.significance
          ? "orientation"
          : "continuity",
      significance: Math.max(
        orientations.significance,
        continuity.significance
      ),
      features: {
        continuity,
        orientations,
      },
    };
  }

  analyzeTextureTransitions(textures) {
    const transitions = [];
    for (let i = 0; i < textures.length - 1; i++) {
      transitions.push({
        complexity: Math.abs(
          textures[i].complexity - textures[i + 1].complexity
        ),
        regularity: Math.abs(
          textures[i].regularity - textures[i + 1].regularity
        ),
        orientation: this.compareTextureOrientations(
          textures[i].orientation,
          textures[i + 1].orientation
        ),
      });
    }
    return transitions;
  }

  calculateTextureOrientationDifference(orientation1, orientation2) {
    if (!orientation1?.dominant || !orientation2?.dominant) return 1;

    const angles1 = orientation1.dominant.map((d) => d.angle);
    const angles2 = orientation2.dominant.map((d) => d.angle);

    let minDiff = Infinity;
    for (const a1 of angles1) {
      for (const a2 of angles2) {
        const diff = Math.abs(a1 - a2);
        minDiff = Math.min(minDiff, Math.min(diff, 360 - diff));
      }
    }

    return minDiff / 180;
  }

  calculateTextureTransitions(textures) {
    const transitions = [];
    for (let i = 0; i < textures.length - 1; i++) {
      transitions.push({
        complexityDiff: Math.abs(
          textures[i].complexity - textures[i + 1].complexity
        ),
        regularityDiff: Math.abs(
          textures[i].regularity - textures[i + 1].regularity
        ),
        orientationDiff: this.calculateTextureOrientationDifference(
          textures[i].orientation,
          textures[i + 1].orientation
        ),
      });
    }
    return transitions;
  }

  compareTextureTransitions(t1, t2) {
    const complexitySimilarity =
      1 - Math.abs(t1.complexityDiff - t2.complexityDiff);
    const regularitySimilarity =
      1 - Math.abs(t1.regularityDiff - t2.regularityDiff);
    const orientationSimilarity =
      1 - Math.abs(t1.orientationDiff - t2.orientationDiff);

    return (
      (complexitySimilarity + regularitySimilarity + orientationSimilarity) / 3
    );
  }

  compareTextureTransitionSegments(segment1, segment2) {
    if (segment1.length !== segment2.length) return 0;

    const similarities = segment1.map((t1, i) => {
      const t2 = segment2[i];
      return this.compareTextureTransitions(t1, t2);
    });

    return mathUtils.calculateMean(similarities);
  }

  evaluateTexturePatternPeriod(transitions, period) {
    const segments = [];
    for (let i = 0; i < transitions.length - period + 1; i += period) {
      segments.push(transitions.slice(i, i + period));
    }

    if (segments.length < 2) {
      return { significance: 0, consistency: 0 };
    }

    // Compare all segments with the first one
    const consistencyScores = segments
      .slice(1)
      .map((segment) =>
        this.compareTextureTransitionSegments(segments[0], segment)
      );

    const consistency = mathUtils.calculateMean(consistencyScores);
    const significance = consistency * (segments.length / transitions.length);

    return { significance, consistency };
  }

  detectRepeatingTexturePattern(transitions) {
    const maxPeriod = Math.floor(transitions.length / 2);
    let bestPeriod = 1;
    let maxSignificance = 0;
    let bestConsistency = 0;

    for (let period = 1; period <= maxPeriod; period++) {
      const { significance, consistency } = this.evaluateTexturePatternPeriod(
        transitions,
        period
      );
      if (significance > maxSignificance) {
        maxSignificance = significance;
        bestPeriod = period;
        bestConsistency = consistency;
      }
    }

    return {
      significance: maxSignificance,
      period: bestPeriod,
      consistency: bestConsistency,
    };
  }

  calculateVariability(values) {
    const mean = mathUtils.calculateMean(values);
    const variance = mathUtils.calculateVariance(values, mean);
    return Math.sqrt(variance) / mean;
  }

  calculateGradientMeasure(differences) {
    if (differences.length < 2) {
      return { significance: 0, consistency: 0, direction: 0 };
    }

    // Check if differences form a consistent trend
    let increasingCount = 0;
    let decreasingCount = 0;

    for (let i = 1; i < differences.length; i++) {
      if (differences[i] > differences[i - 1]) increasingCount++;
      else if (differences[i] < differences[i - 1]) decreasingCount++;
    }

    const totalComparisons = differences.length - 1;
    const consistency =
      Math.max(increasingCount, decreasingCount) / totalComparisons;
    const direction = increasingCount >= decreasingCount ? 1 : -1;
    const significance =
      consistency * (1 - this.calculateVariability(differences));

    return {
      significance,
      consistency,
      direction,
    };
  }

  detectTextureGradient(transitions) {
    const complexityGradient = this.calculateGradientMeasure(
      transitions.map((t) => t.complexityDiff)
    );
    const regularityGradient = this.calculateGradientMeasure(
      transitions.map((t) => t.regularityDiff)
    );
    const orientationGradient = this.calculateGradientMeasure(
      transitions.map((t) => t.orientationDiff)
    );

    const significance =
      complexityGradient.significance * 0.4 +
      regularityGradient.significance * 0.3 +
      orientationGradient.significance * 0.3;

    return {
      significance,
      components: {
        complexity: complexityGradient,
        regularity: regularityGradient,
        orientation: orientationGradient,
      },
    };
  }

  analyzeTexturePattern(cells) {
    const textureFeatures = cells
      .map((cell) => cell.texture)
      .filter((texture) => texture);

    if (textureFeatures.length < 2) {
      return { type: "none", significance: 0 };
    }

    // Calculate texture transitions
    const transitions = this.calculateTextureTransitions(textureFeatures);

    // Detect patterns
    const repeating = this.detectRepeatingTexturePattern(transitions);
    const gradient = this.detectTextureGradient(transitions);

    return {
      type:
        repeating.significance > gradient.significance
          ? "repeating"
          : "gradient",
      significance: Math.max(repeating.significance, gradient.significance),
      features: {
        repeating,
        gradient,
      },
    };
  }

  analyzeRowPattern(row) {
    if (!row || !Array.isArray(row)) {
      return {
        type: "none",
        significance: 0,
        features: {},
      };
    }

    const colorPattern = this.analyzeColorPattern(row);
    const edgePattern = this.analyzeEdgePattern(row);
    const texturePattern = this.analyzeTexturePattern(row);

    const patterns = [colorPattern, edgePattern, texturePattern].filter(
      (p) => p.significance > 0.3
    );

    if (patterns.length === 0) {
      return {
        type: "none",
        significance: 0,
        features: {},
      };
    }

    // Find most significant pattern
    const dominantPattern = patterns.reduce((a, b) =>
      a.significance > b.significance ? a : b
    );

    return {
      ...dominantPattern,
      features: {
        color: colorPattern,
        edge: edgePattern,
        texture: texturePattern,
      },
    };
  }

  analyzeColumnPattern(column) {
    // Similar to analyzeRowPattern but for vertical patterns
    return this.analyzeRowPattern(column);
  }

  detectGridPatterns(grid) {
    if (!grid || !Array.isArray(grid) || grid.length === 0) {
      return [];
    }

    const patterns = [];
    const gridSize = grid.length;

    // Detect horizontal patterns
    for (let y = 0; y < gridSize; y++) {
      const rowPattern = this.analyzeRowPattern(grid[y]);
      if (rowPattern.significance > 0.5) {
        patterns.push({
          type: "horizontal",
          row: y,
          ...rowPattern,
        });
      }
    }

    // Detect vertical patterns
    for (let x = 0; x < gridSize; x++) {
      const column = grid.map((row) => row[x]);
      const colPattern = this.analyzeColumnPattern(column);
      if (colPattern.significance > 0.5) {
        patterns.push({
          type: "vertical",
          column: x,
          ...colPattern,
        });
      }
    }

    return patterns;
  }

  compareTexturePatterns(patterns1, patterns2) {
    if (!patterns1 || !patterns2) return 0;

    const uniformitySimilarity =
      1 - Math.abs(patterns1.uniformity - patterns2.uniformity);

    const complexitySimilarity =
      1 - Math.abs(patterns1.complexity - patterns2.complexity);

    const regularitySimilarity =
      1 - Math.abs(patterns1.regularity - patterns2.regularity);

    return (
      uniformitySimilarity * 0.4 +
      complexitySimilarity * 0.3 +
      regularitySimilarity * 0.3
    );
  }

  compareDominantOrientations(dom1, dom2) {
    if (!dom1 || !dom2 || !Array.isArray(dom1) || !Array.isArray(dom2)) {
      return 0;
    }

    let maxSimilarity = 0;
    for (const o1 of dom1) {
      for (const o2 of dom2) {
        const similarity = this.calculateOrientationSimilarity(o1, o2);
        maxSimilarity = Math.max(maxSimilarity, similarity);
      }
    }

    return maxSimilarity;
  }

  compareOrientations(orientation1, orientation2) {
    if (!orientation1 || !orientation2) return 0;

    // Compare dominant orientations
    const dominantSimilarity = this.compareDominantOrientations(
      orientation1.dominant || [],
      orientation2.dominant || []
    );

    // Compare coherence
    const coherenceSimilarity =
      1 -
      Math.abs((orientation1.coherence || 0) - (orientation2.coherence || 0));

    return dominantSimilarity * 0.6 + coherenceSimilarity * 0.4;
  }

  // Add cache key generation:
  generateCacheKey(buffer) {
    const hash = crypto.createHash("sha256");
    hash.update(buffer);
    return `signature_${hash.digest("hex")}`;
  }

  async extractMetadata(buffer) {
    try {
      const metadata = await sharp(buffer).metadata();
      return {
        width: metadata.width,
        height: metadata.height,
        format: metadata.format,
        space: metadata.space,
        channels: metadata.channels,
        density: metadata.density,
        hasAlpha: metadata.hasAlpha,
        orientation: metadata.orientation,
      };
    } catch (error) {
      console.error("Metadata extraction failed:", error);
      return {
        width: 0,
        height: 0,
        format: "unknown",
        space: "unknown",
        channels: 0,
      };
    }
  }

  // Modify the main generateSignature method
  async generateEdgeFeatures(buffer, metadata) {
    const edges = await this.extractEdgeMap(buffer);
    if (!edges || !edges.edges) {
      return this.getDefaultEdgeFeatures();
    }

    return {
      edges: edges.edges,
      features: await this.extractEdgeFeatures(edges),
      quality: this.assessEdgeQuality(edges),
    };
  }

  async generateHash(buffer, metadata) {
    try {
      const { image } = await this.prepareValidBuffer(buffer);

      // Ensure consistent size for hash generation
      const resized = await image
        .resize(this.config.HASH.DCT_SIZE, this.config.HASH.DCT_SIZE, {
          fit: "fill",
          kernel: sharp.kernel.lanczos3,
        })
        .raw()
        .toBuffer();

      return resized;
    } catch (error) {
      console.error("Hash preparation failed:", error);
      return null;
    }
  }

  // Main signature generation with proper buffer handling
  async generateSignature(buffer) {
    try {
      console.log("Starting signature generation...");

      if (!Buffer.isBuffer(buffer)) {
        throw new Error("Invalid input: Buffer expected");
      }

      console.log("Processing and validating buffer...");
      // Process and validate the buffer
      const { buffer: validBuffer, metadata } =
        await this.validateAndProcessImage(buffer);
      console.log("Buffer validated successfully. Processing features...");

      // Extract features with timeout protection and better logging
      console.log("Starting color feature extraction...");
      const colorFeatures = await this._safeExtract(async () => {
        return await this.extractColorFeatures(validBuffer);
      });
      console.log("Color features extracted");

      console.log("Starting edge feature extraction...");
      const edgeFeatures = await this._safeExtract(async () => {
        return await this.extractEdgeFeatures(validBuffer);
      });
      console.log("Edge features extracted");

      console.log("Starting texture feature extraction...");
      const textureFeatures = await this._safeExtract(async () => {
        return await this.extractTextureFeatures(validBuffer);
      });
      console.log("Texture features extracted");

      console.log("Starting perceptual hash generation...");
      const perceptualHashes = await this._safeExtract(async () => {
        return await this.generatePerceptualHashes(validBuffer);
      });
      console.log("Perceptual hashes generated");

      // Generate spatial verification
      console.log("Starting spatial verification...");
      const spatialFeatures = await this._safeExtract(async () => {
        return await this.generateSpatialVerification(validBuffer, {
          colorFeatures,
          edgeFeatures,
          textureFeatures,
        });
      });
      console.log("Spatial verification completed");

      // Construct signature
      console.log("Constructing final signature...");
      const signature = {
        version: "2.0",
        timestamp: Date.now(),
        colors: colorFeatures || this.getDefaultColorFeatures(),
        edges: edgeFeatures || this.getDefaultEdgeFeatures(),
        textures: textureFeatures || this.getDefaultTextureFeatures(),
        hashes: perceptualHashes || this.getDefaultHashFeatures(),
        spatial: spatialFeatures || this.getDefaultSpatialFeatures(),
        metadata: await this.extractMetadata(buffer),
      };

      // Calculate quality metrics
      console.log("Calculating quality metrics...");
      signature.quality = this.calculateSignatureQuality(signature);

      console.log("Signature generation completed successfully");
      return signature;
    } catch (error) {
      console.error("Detailed signature generation error:", {
        message: error.message,
        stack: error.stack,
        name: error.name,
      });
      throw new Error(`Signature generation failed: ${error.message}`);
    }
  }

  async _generateSignatureWithOptimization(buffer) {
    // Optimize the input image
    buffer = await this._optimizeImageBuffer(buffer);

    // Extract features with timeout protection
    const [colorFeatures, edgeFeatures, textureFeatures, perceptualHashes] =
      await Promise.all([
        this._safeExtract(this.extractColorFeatures.bind(this), buffer),
        this._safeExtract(this.extractEdgeFeatures.bind(this), buffer),
        this._safeExtract(this.extractTextureFeatures.bind(this), buffer),
        this._safeExtract(this.generatePerceptualHashes.bind(this), buffer),
      ]);

    // Generate spatial verification
    const spatialVerification = await this._safeExtract(
      async () =>
        this.generateSpatialVerification(buffer, {
          colorFeatures,
          edgeFeatures,
          textureFeatures,
        }),
      buffer
    );

    // Combine all features into comprehensive signature
    const signature = {
      version: "2.0",
      timestamp: Date.now(),
      colors: colorFeatures,
      edges: edgeFeatures,
      textures: textureFeatures,
      hashes: perceptualHashes,
      spatialVerification,
      metadata: await this.extractMetadata(buffer),
    };

    // Calculate quality metrics
    signature.quality = this.calculateSignatureQuality(signature);

    return signature;
  }

  async verifyBufferIntegrity(buffer) {
    if (!Buffer.isBuffer(buffer)) {
      throw new Error("Invalid buffer type");
    }

    try {
      await sharp(buffer, { failOnError: true }).metadata();
      return true;
    } catch (error) {
      console.error("Buffer integrity check failed:", error);
      return false;
    }
  }

  async _processWithSharp(buffer, operation) {
    try {
      const { buffer: validBuffer, metadata } =
        await this.validateAndPrepareBuffer(buffer);
      const image = sharp(validBuffer, { failOnError: false });
      return await operation(image, metadata);
    } catch (error) {
      throw new Error(`Sharp operation failed: ${error.message}`);
    }
  }

  updateClusterCenter(cluster) {
    if (!cluster || !cluster.points || cluster.points.length === 0) {
      return;
    }

    const sum = cluster.points.reduce(
      (acc, point) => ({
        r: acc.r + point.r,
        g: acc.g + point.g,
        b: acc.b + point.b,
        a: acc.a + (point.a || 255),
      }),
      { r: 0, g: 0, b: 0, a: 0 }
    );

    const count = cluster.points.length;
    cluster.center = {
      r: Math.round(sum.r / count),
      g: Math.round(sum.g / count),
      b: Math.round(sum.b / count),
      a: Math.round(sum.a / count),
    };
  }

  // Update feature extraction methods to use the wrapper
  async findDominantColors(pixels) {
    const maxColors = 8;
    const clusters = [];
    const processed = new Set();

    // Process pixels in chunks
    for (const pixel of pixels) {
      if (clusters.length >= maxColors) break;

      const key = `${pixel.r},${pixel.g},${pixel.b}`;
      if (processed.has(key)) continue;
      processed.add(key);

      let newCluster = true;
      for (const cluster of clusters) {
        if (this.isColorSimilar(pixel, cluster.center)) {
          cluster.points.push(pixel);
          this.updateClusterCenter(cluster);
          newCluster = false;
          break;
        }
      }

      if (newCluster && clusters.length < maxColors) {
        clusters.push({
          center: { ...pixel },
          points: [pixel],
        });
      }

      // Yield to event loop periodically
      if (processed.size % 1000 === 0) {
        await new Promise((resolve) => setTimeout(resolve, 0));
      }
    }

    // Calculate populations and sort by size
    const total = clusters.reduce(
      (sum, cluster) => sum + cluster.points.length,
      0
    );
    return clusters
      .map((cluster) => ({
        rgb: cluster.center,
        population: cluster.points.length / total,
        confidence: this.calculateClusterConfidence(cluster),
      }))
      .sort((a, b) => b.population - a.population);
  }

  async analyzeColorHarmonies(dominantColors) {
    return {
      complementary: this.findComplementaryColors(dominantColors),
      analogous: this.findAnalogousColors(dominantColors),
      triadic: this.findTriadicHarmonies(dominantColors),
      splitComplementary: this.findSplitComplementary(dominantColors),
      monochromatic: this.findMonochromaticVariations(dominantColors),
      quality: this.assessHarmonyQuality(dominantColors),
    };
  }

  async calculatePerceptualColorMetrics(data, info, dominantColors) {
    return {
      colorfulness: this.calculateColorfulness(data, info),
      contrast: this.calculatePerceptualContrast(dominantColors),
      saturation: this.calculateOverallSaturation(data, info),
      complexity: this.calculateColorComplexity(dominantColors),
    };
  }

  // Color space conversion utilities
  async convertToLAB(data, info) {
    const colors = [];
    const stride = info.channels;

    for (let i = 0; i < data.length; i += stride) {
      const rgb = {
        r: data[i],
        g: data[i + 1],
        b: data[i + 2],
      };

      colors.push(this.rgbToLAB(rgb));
    }

    return colors;
  }

  calculateWaveletEntropy(coefficients) {
    if (!coefficients) return 0;

    let totalEntropy = 0;
    let count = 0;

    for (const band in coefficients) {
      if (coefficients[band] && coefficients[band].length) {
        const values = Array.from(coefficients[band]);
        totalEntropy += this.calculateEntropy(values);
        count++;
      }
    }

    return count > 0 ? totalEntropy / count : 0;
  }

  calculateWaveletPersistence(coefficients) {
    if (!coefficients) return 0;

    const bands = ["LL", "LH", "HL", "HH"];
    let persistence = 0;
    let count = 0;

    for (const band of bands) {
      if (coefficients[band] && coefficients[band].length) {
        const values = Array.from(coefficients[band]);
        const max = Math.max(...values);
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        persistence += max / (mean + 1e-10);
        count++;
      }
    }

    return count > 0 ? persistence / count : 0;
  }

  assessScaleStability(hash) {
    const scaleFactors = [0.9, 0.95, 1.05, 1.1];
    const similarities = scaleFactors.map((scale) => {
      const scaled = this.simulateScaling(hash, scale);
      return this.calculateHashSimilarity(hash, scaled);
    });

    return (
      similarities.reduce((acc, val) => acc + val, 0) / similarities.length
    );
  }

  simulateTranslation(hash, offset) {
    const size = Math.sqrt(hash.length * 8);
    const translated = new Uint8Array(hash.length);

    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const newX = x + offset;
        const newY = y;

        if (newX >= 0 && newX < size) {
          const oldBit = this.getBit(hash, x, y, size);
          this.setBit(translated, newX, newY, size, oldBit);
        }
      }
    }

    return translated;
  }

  assessTranslationStability(hash) {
    const offsets = [-2, -1, 1, 2];
    const similarities = offsets.map((offset) => {
      const translated = this.simulateTranslation(hash, offset);
      return this.calculateHashSimilarity(hash, translated);
    });

    return (
      similarities.reduce((acc, val) => acc + val, 0) / similarities.length
    );
  }

  async extractCellEdges(edgeMap, x, y, width, height) {
    try {
      if (!edgeMap || !edgeMap.edges) {
        return [];
      }

      const cellEdges = [];
      const cellWidth = Math.floor(width / 8);
      const cellHeight = Math.floor(height / 8);
      const startX = x * cellWidth;
      const startY = y * cellHeight;
      const endX = Math.min(startX + cellWidth, width);
      const endY = Math.min(startY + cellHeight, height);

      for (let py = startY; py < endY; py++) {
        for (let px = startX; px < endX; px++) {
          const idx = py * width + px;
          if (edgeMap.edges[idx] > 0) {
            cellEdges.push(edgeMap.edges[idx]);
          }
        }
      }

      return cellEdges;
    } catch (error) {
      console.error("Cell edge extraction failed:", error);
      return [];
    }
  }

  // Add missing methods
  calculateCellEdgeStrength(edgeMap, x, y, width, height) {
    try {
      if (!edgeMap || !edgeMap.edges) return 0;

      const cellEdges = this.extractCellEdges(edgeMap, x, y, width, height);
      if (!cellEdges || !cellEdges.length) return 0;

      return cellEdges.reduce((sum, edge) => sum + edge, 0) / cellEdges.length;
    } catch (error) {
      console.error("Cell edge strength calculation failed:", error);
      return 0;
    }
  }

  analyzeRegionRelationships(regions) {
    try {
      if (!regions || !Array.isArray(regions)) {
        return [];
      }

      const relationships = [];
      for (let i = 0; i < regions.length; i++) {
        for (let j = i + 1; j < regions.length; j++) {
          relationships.push({
            region1: i,
            region2: j,
            relationship: this.calculateRegionRelationship(
              regions[i],
              regions[j]
            ),
          });
        }
      }
      return relationships;
    } catch (error) {
      console.error("Region relationship analysis failed:", error);
      return [];
    }
  }

  combineTextureFeatures(features) {
    if (!features) return {};

    const combined = {
      contrast: 0,
      coarseness: 0,
      directionality: 0,
      regularity: 0,
    };

    let validFeatures = 0;

    // GLCM contribution
    if (features.glcm) {
      combined.contrast += features.glcm.contrast || 0;
      combined.regularity += features.glcm.energy || 0;
      validFeatures++;
    }

    // LBP contribution
    if (features.lbp && features.lbp.histogram) {
      combined.coarseness += this.calculateLBPCoarseness(
        features.lbp.histogram
      );
      validFeatures++;
    }

    // Gabor contribution
    if (features.gabor && features.gabor.features) {
      combined.directionality += this.calculateGaborDirectionality(
        features.gabor.features
      );
      validFeatures++;
    }

    // Wavelet contribution
    if (features.wavelet && features.wavelet.coefficients) {
      combined.coarseness += this.calculateWaveletCoarseness(
        features.wavelet.coefficients
      );
      validFeatures++;
    }

    // Normalize if we have valid features
    if (validFeatures > 0) {
      for (let key in combined) {
        combined[key] /= validFeatures;
      }
    }

    return combined;
  }

  calculateGlobalGaborStatistics(features) {
    if (!features || !Array.isArray(features)) {
      return {
        mean: 0,
        std: 0,
        energy: 0,
      };
    }

    const values = features.flatMap((f) =>
      f.features ? [f.features.mean, f.features.energy] : []
    );

    return {
      mean: this.calculateMean(values),
      std: this.calculateStd(values),
      energy: values.reduce((sum, val) => sum + val * val, 0) / values.length,
    };
  }

  analyzeWaveletEnergy(coefficients) {
    try {
      if (!coefficients || typeof coefficients !== "object") {
        return {
          distribution: new Map(),
          score: 0,
          concentration: 0,
        };
      }

      const energyByLevel = new Map();
      let totalEnergy = 0;

      // Safely calculate energy for each subband
      Object.entries(coefficients).forEach(([band, values]) => {
        if (Array.isArray(values) || ArrayBuffer.isView(values)) {
          const energy = Array.from(values).reduce((sum, c) => sum + c * c, 0);
          energyByLevel.set(band, energy);
          totalEnergy += energy;
        }
      });

      // Normalize energies
      const normalizedEnergies = new Map();
      energyByLevel.forEach((energy, band) => {
        normalizedEnergies.set(
          band,
          totalEnergy > 0 ? energy / totalEnergy : 0
        );
      });

      return {
        distribution: normalizedEnergies,
        score: this.calculateEnergyScore(normalizedEnergies),
        concentration: this.calculateEnergyConcentration(normalizedEnergies),
      };
    } catch (error) {
      console.error("Wavelet energy analysis failed:", error);
      return {
        distribution: new Map(),
        score: 0,
        concentration: 0,
      };
    }
  }

  // Fix #5: Enhanced spatial analysis with missing function implementations

  // Add missing cell color extraction function
  extractCellColors(colorLayout, x, y, width, height) {
    if (!colorLayout || !colorLayout.data) {
      return [];
    }

    try {
      const colors = [];
      const cellWidth = Math.floor(width);
      const cellHeight = Math.floor(height);
      const channels = colorLayout.channels || 4;

      for (let dy = 0; dy < cellHeight; dy++) {
        for (let dx = 0; dx < cellWidth; dx++) {
          const px = x + dx;
          const py = y + dy;

          if (px < colorLayout.width && py < colorLayout.height) {
            const idx = (py * colorLayout.width + px) * channels;
            colors.push({
              r: colorLayout.data[idx] || 0,
              g: colorLayout.data[idx + 1] || 0,
              b: colorLayout.data[idx + 2] || 0,
              a: channels > 3 ? colorLayout.data[idx + 3] : 255,
            });
          }
        }
      }

      return colors;
    } catch (error) {
      console.error("Cell color extraction failed:", error);
      return [];
    }
  }

  calculateEnergyScore(normalizedEnergies) {
    if (!normalizedEnergies || typeof normalizedEnergies.get !== "function") {
      return 0;
    }

    const weights = {
      LL: 0.4,
      LH: 0.2,
      HL: 0.2,
      HH: 0.2,
    };

    let score = 0;
    for (const [band, energy] of normalizedEnergies) {
      if (weights[band]) {
        score += energy * weights[band];
      }
    }
    return score;
  }

  calculateEnergyConcentration(normalizedEnergies) {
    if (!normalizedEnergies || typeof normalizedEnergies.get !== "function") {
      return 0;
    }

    const values = Array.from(normalizedEnergies.values());
    const mean = this.calculateMean(values);
    if (mean === 0) return 0;

    const std = this.calculateStd(values);
    return 1 - std / mean;
  }

  // Additional required helper methods
  calculateLBPCoarseness(histogram) {
    if (!histogram || !Array.isArray(histogram)) return 0;

    let coarseness = 0;
    for (let i = 0; i < histogram.length; i++) {
      if (histogram[i] > 0) {
        coarseness += histogram[i] * Math.log2(i + 1);
      }
    }
    return coarseness;
  }

  calculateGaborDirectionality(features) {
    if (!features || !Array.isArray(features)) return 0;

    let maxResponse = 0;
    let totalResponse = 0;

    features.forEach((feature) => {
      if (feature && feature.energy) {
        maxResponse = Math.max(maxResponse, feature.energy);
        totalResponse += feature.energy;
      }
    });

    return totalResponse > 0 ? maxResponse / totalResponse : 0;
  }

  calculateWaveletCoarseness(coefficients) {
    if (!coefficients || typeof coefficients !== "object") return 0;

    let totalEnergy = 0;
    let weightedEnergy = 0;
    let level = 1;

    for (const [band, values] of Object.entries(coefficients)) {
      if (Array.isArray(values) || ArrayBuffer.isView(values)) {
        const energy = values.reduce((sum, c) => sum + c * c, 0);
        totalEnergy += energy;
        weightedEnergy += energy * level;
      }
      level++;
    }

    return totalEnergy > 0 ? weightedEnergy / totalEnergy : 0;
  }

  rgbToLAB(rgb) {
    // RGB to XYZ
    let r = rgb.r / 255;
    let g = rgb.g / 255;
    let b = rgb.b / 255;

    r = r > 0.04045 ? Math.pow((r + 0.055) / 1.055, 2.4) : r / 12.92;
    g = g > 0.04045 ? Math.pow((g + 0.055) / 1.055, 2.4) : g / 12.92;
    b = b > 0.04045 ? Math.pow((b + 0.055) / 1.055, 2.4) : b / 12.92;

    const x = (r * 0.4124 + g * 0.3576 + b * 0.1805) * 100;
    const y = (r * 0.2126 + g * 0.7152 + b * 0.0722) * 100;
    const z = (r * 0.0193 + g * 0.1192 + b * 0.9505) * 100;

    // XYZ to LAB
    const xn = 95.047;
    const yn = 100.0;
    const zn = 108.883;

    const xr = x / xn;
    const yr = y / yn;
    const zr = z / zn;

    const fx = xr > 0.008856 ? Math.pow(xr, 1 / 3) : 7.787 * xr + 16 / 116;
    const fy = yr > 0.008856 ? Math.pow(yr, 1 / 3) : 7.787 * yr + 16 / 116;
    const fz = zr > 0.008856 ? Math.pow(zr, 1 / 3) : 7.787 * zr + 16 / 116;

    return {
      l: 116 * fy - 16,
      a: 500 * (fx - fy),
      b: 200 * (fy - fz),
    };
  }

  labToRGB(lab) {
    // LAB to XYZ
    const y = (lab.l + 16) / 116;
    const x = lab.a / 500 + y;
    const z = y - lab.b / 200;

    const xn = 95.047;
    const yn = 100.0;
    const zn = 108.883;

    const x3 = Math.pow(x, 3);
    const z3 = Math.pow(z, 3);
    const y3 = Math.pow(y, 3);

    const xr = x3 > 0.008856 ? x3 : (x - 16 / 116) / 7.787;
    const yr = y3 > 0.008856 ? y3 : (y - 16 / 116) / 7.787;
    const zr = z3 > 0.008856 ? z3 : (z - 16 / 116) / 7.787;

    const x_ = xr * xn;
    const y_ = yr * yn;
    const z_ = zr * zn;

    // XYZ to RGB
    let r = x_ * 3.2406 + y_ * -1.5372 + z_ * -0.4986;
    let g = x_ * -0.9689 + y_ * 1.8758 + z_ * 0.0415;
    let b = x_ * 0.0557 + y_ * -0.204 + z_ * 1.057;

    r = r > 0.0031308 ? 1.055 * Math.pow(r, 1 / 2.4) - 0.055 : 12.92 * r;
    g = g > 0.0031308 ? 1.055 * Math.pow(g, 1 / 2.4) - 0.055 : 12.92 * g;
    b = b > 0.0031308 ? 1.055 * Math.pow(b, 1 / 2.4) - 0.055 : 12.92 * b;

    return {
      r: Math.max(0, Math.min(255, Math.round(r * 255))),
      g: Math.max(0, Math.min(255, Math.round(g * 255))),
      b: Math.max(0, Math.min(255, Math.round(b * 255))),
    };
  }

  // Default feature generators

  extractCellTextureFeatures(buffer) {
    try {
      if (!buffer) {
        return this.getDefaultTextureFeatures();
      }

      return {
        glcm: [],
        lbp: [],
        gabor: [],
        statistics: {},
        quality: {
          overall: 0,
          components: {
            glcm: 0,
            lbp: 0,
            gabor: 0,
          },
        },
      };
    } catch (error) {
      console.error("Cell texture feature extraction failed:", error);
      return this.getDefaultTextureFeatures();
    }
  }

  getDefaultCellFeatures() {
    return {
      color: {
        dominant: null,
        distribution: new Map(),
        coherence: 0,
        importance: 0,
      },
      edges: {
        density: 0,
        strength: 0,
        continuity: 0,
        importance: 0,
      },
      texture: this.getDefaultCellTextureFeatures(),
      importance: 0,
      consistency: 0,
      distinctiveness: 0,
    };
  }

  async extractCellFeatures(buffer, x, y, width, height) {
    try {
      if (!buffer) {
        throw new Error("No buffer provided for cell feature extraction");
      }

      const [colorFeatures, edgeFeatures, textureFeatures] = await Promise.all([
        this.extractCellColorFeatures(buffer, x, y, width, height),
        this.extractCellEdgeFeatures(buffer, x, y, width, height),
        this.extractCellTextureFeatures(buffer, x, y, width, height),
      ]);

      return {
        color: colorFeatures,
        edges: edgeFeatures,
        texture: textureFeatures,
        combined: this.combineCellFeatures({
          color: colorFeatures,
          edges: edgeFeatures,
          texture: textureFeatures,
        }),
      };
    } catch (error) {
      console.error("Cell feature extraction failed:", error);
      return this.getDefaultCellFeatures();
    }
  }

  getDefaultCellTextureFeatures() {
    return {
      glcm: [],
      lbp: [],
      gabor: [],
      statistics: {},
      quality: {
        overall: 0,
        components: {
          glcm: 0,
          lbp: 0,
          gabor: 0,
        },
      },
    };
  }

  getDefaultCellEdgeFeatures() {
    return {
      density: 0,
      strength: 0,
      orientation: {
        angle: 0,
        magnitude: 0,
      },
      continuity: 0,
    };
  }

  combineCellFeatures(features) {
    const weights = {
      color: 0.4,
      edges: 0.3,
      texture: 0.3,
    };

    return {
      importance: Object.entries(features).reduce(
        (sum, [key, value]) => sum + (value?.importance || 0) * weights[key],
        0
      ),
      consistency: Object.entries(features).reduce(
        (sum, [key, value]) => sum + (value?.consistency || 0) * weights[key],
        0
      ),
      distinctiveness: Object.entries(features).reduce(
        (sum, [key, value]) =>
          sum + (value?.distinctiveness || 0) * weights[key],
        0
      ),
    };
  }

  // Updated image scales generation
  async generateImageScales(buffer) {
    try {
      if (!buffer || !Buffer.isBuffer(buffer)) {
        throw new Error("Invalid buffer input");
      }

      const metadata = await sharp(buffer).metadata();
      if (!metadata || !metadata.width || !metadata.height) {
        throw new Error("Invalid image metadata");
      }

      const scales = [buffer];
      let currentBuffer = buffer;
      const minDimension = Math.min(metadata.width, metadata.height);
      const maxScales = Math.floor(Math.log2(minDimension / 32));

      for (let i = 0; i < maxScales; i++) {
        currentBuffer = await sharp(currentBuffer, { failOnError: false })
          .resize(
            Math.round(metadata.width / Math.pow(2, i + 1)),
            Math.round(metadata.height / Math.pow(2, i + 1)),
            {
              kernel: sharp.kernel.lanczos3,
              fastShrinkOnLoad: true,
            }
          )
          .toBuffer();
        scales.push(currentBuffer);
      }

      return scales;
    } catch (error) {
      console.error("Scale generation failed:", error);
      return [buffer]; // Return original buffer as fallback
    }
  }

  async analyzeEdgeScale(buffer) {
    // Apply advanced edge detection methods
    const [cannyEdges, sobelGradients, laplacian] = await Promise.all([
      this.applyCannyEdgeDetection(buffer),
      this.applySobelOperator(buffer),
      this.applyLaplacian(buffer),
    ]);

    // Combine edge detection results
    const combinedEdges = this.combineEdgeDetectors(
      cannyEdges,
      sobelGradients,
      laplacian
    );

    // Extract edge features
    return {
      edges: combinedEdges,
      magnitude: await this.calculateEdgeMagnitude(combinedEdges),
      direction: await this.calculateEdgeDirections(combinedEdges),
      features: this.extractEdgeFeatures(combinedEdges),
      quality: this.assessEdgeDetectionQuality(combinedEdges),
    };
  }

  async applyCannyEdgeDetection(buffer) {
    const { data, info } = await sharp(buffer)
      .grayscale()
      .raw()
      .toBuffer({ resolveWithObject: true });

    // Gaussian smoothing
    const smoothed = await this.applyGaussianBlur(
      data,
      info.width,
      info.height
    );

    // Gradient calculation
    const [magnitude, direction] = this.calculateGradients(
      smoothed,
      info.width
    );

    // Non-maximum suppression
    const suppressed = this.applyNonMaxSuppression(
      magnitude,
      direction,
      info.width
    );

    // Double thresholding and edge tracking
    return this.performHysteresisThresholding(
      suppressed,
      info.width,
      this.config.EDGE.CANNY_LOW,
      this.config.EDGE.CANNY_HIGH
    );
  }

  async applySobelOperator(buffer) {
    const { data, info } = await sharp(buffer)
      .grayscale()
      .raw()
      .toBuffer({ resolveWithObject: true });

    const width = info.width;
    const height = info.height;
    const gradientX = new Float32Array(data.length);
    const gradientY = new Float32Array(data.length);

    // Sobel kernels
    const sobelX = [
      [-1, 0, 1],
      [-2, 0, 2],
      [-1, 0, 1],
    ];
    const sobelY = [
      [-1, -2, -1],
      [0, 0, 0],
      [1, 2, 1],
    ];

    // Apply Sobel operators
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        let sumX = 0;
        let sumY = 0;

        for (let i = -1; i <= 1; i++) {
          for (let j = -1; j <= 1; j++) {
            const idx = (y + i) * width + (x + j);
            sumX += data[idx] * sobelX[i + 1][j + 1];
            sumY += data[idx] * sobelY[i + 1][j + 1];
          }
        }

        const idx = y * width + x;
        gradientX[idx] = sumX;
        gradientY[idx] = sumY;
      }
    }

    return { gradientX, gradientY };
  }

  async applyLaplacian(buffer) {
    const { data, info } = await sharp(buffer)
      .grayscale()
      .raw()
      .toBuffer({ resolveWithObject: true });

    const width = info.width;
    const height = info.height;
    const laplacian = new Float32Array(data.length);

    // Laplacian kernel
    const kernel = [
      [0, 1, 0],
      [1, -4, 1],
      [0, 1, 0],
    ];

    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        let sum = 0;

        for (let i = -1; i <= 1; i++) {
          for (let j = -1; j <= 1; j++) {
            const idx = (y + i) * width + (x + j);
            sum += data[idx] * kernel[i + 1][j + 1];
          }
        }

        laplacian[y * width + x] = sum;
      }
    }

    return laplacian;
  }

  combineEdgeDetectors(cannyEdges, sobelGradients, laplacian) {
    const combined = new Float32Array(cannyEdges.length);
    const { gradientX, gradientY } = sobelGradients;

    for (let i = 0; i < combined.length; i++) {
      // Calculate Sobel magnitude
      const sobelMagnitude = Math.sqrt(
        gradientX[i] * gradientX[i] + gradientY[i] * gradientY[i]
      );

      // Weighted combination of edge detectors
      combined[i] =
        cannyEdges[i] * 0.5 +
        (sobelMagnitude / 255) * 0.3 +
        (Math.abs(laplacian[i]) / 255) * 0.2;
    }

    return combined;
  }

  async extractPrimaryEdges(buffer) {
    const { data: edges, info } = await this.applyCannyEdgeDetection(buffer);
    const segments = this.extractEdgeSegments(edges, info.width);

    return {
      edges,
      segments: this.filterSignificantSegments(segments),
      strength: this.calculateSegmentStrength(segments),
      orientation: this.calculateSegmentOrientations(segments),
      linkage: this.analyzeSegmentLinkage(segments),
    };
  }

  extractEdgeSegments(edges, width) {
    const segments = [];
    const visited = new Set();
    const height = edges.length / width;

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = y * width + x;
        if (edges[idx] > 0 && !visited.has(idx)) {
          const segment = this.traceEdgeSegment(edges, x, y, width, visited);
          if (segment.points.length >= this.config.EDGE.MIN_EDGE_LENGTH) {
            segments.push(segment);
          }
        }
      }
    }

    return segments;
  }

  traceEdgeSegment(edges, startX, startY, width, visited) {
    const segment = {
      points: [],
      strength: 0,
    };

    const stack = [[startX, startY]];
    while (stack.length > 0) {
      const [x, y] = stack.pop();
      const idx = y * width + x;

      if (visited.has(idx)) continue;
      visited.add(idx);

      segment.points.push([x, y]);
      segment.strength += edges[idx];

      // Check 8-connected neighbors
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue;

          const nx = x + dx;
          const ny = y + dy;
          const nidx = ny * width + nx;

          if (edges[nidx] > 0 && !visited.has(nidx)) {
            stack.push([nx, ny]);
          }
        }
      }
    }

    return segment;
  }

  harrisCornerDetection(data, width, height) {
    const corners = [];
    const windowSize = 3;
    const k = 0.04; // Harris detector free parameter

    // Calculate gradients
    const { gradientX, gradientY } = this.calculateImageGradients(data, width);

    for (let y = windowSize; y < height - windowSize; y++) {
      for (let x = windowSize; x < width - windowSize; x++) {
        const [sumXX, sumYY, sumXY] = this.calculateHarrisMatrix(
          gradientX,
          gradientY,
          x,
          y,
          width,
          windowSize
        );

        // Calculate corner response
        const det = sumXX * sumYY - sumXY * sumXY;
        const trace = sumXX + sumYY;
        const response = det - k * trace * trace;

        if (response > 0.1) {
          // Threshold for corner detection
          corners.push({ x, y, response });
        }
      }
    }

    return this.nonMaximaSuppression(corners, width, height);
  }

  async analyzeEdgeTopology(components) {
    const topology = {
      graph: this.constructEdgeGraph(components),
      connectivity: this.analyzeGraphConnectivity(components),
      hierarchy: this.buildEdgeHierarchy(components),
      relationships: this.analyzeEdgeRelationships(components),
      metrics: this.calculateTopologyMetrics(components),
    };

    return {
      ...topology,
      quality: this.assessTopologyQuality(topology),
    };
  }

  buildEdgeHierarchy(components) {
    return {
      levels: this.constructHierarchicalLevels(components),
      relations: this.analyzeHierarchicalRelations(components),
      structure: this.analyzeHierarchicalStructure(components),
    };
  }

  constructHierarchicalLevels(components) {
    const { primary, secondary, junctions, corners } = components;

    return [
      {
        level: 0,
        type: "primary",
        elements: this.processElements(primary.segments),
        metrics: this.calculateLevelMetrics(primary.segments),
      },
      {
        level: 1,
        type: "secondary",
        elements: this.processElements(secondary.segments),
        metrics: this.calculateLevelMetrics(secondary.segments),
      },
      {
        level: 2,
        type: "features",
        elements: [
          ...this.processJunctions(junctions.points),
          ...this.processCorners(corners.points),
        ],
        metrics: this.calculateFeatureMetrics(junctions.points, corners.points),
      },
    ];
  }

  processElements(segments) {
    return segments.map((segment) => ({
      ...segment,
      importance: this.calculateElementImportance(segment),
      relations: this.identifyElementRelations(segment),
    }));
  }

  calculateElementImportance(element) {
    const lengthFactor = Math.min(1, element.points.length / 100);
    const strengthFactor = element.strength / 255;
    const continuityFactor = this.calculateContinuityFactor(element);

    return lengthFactor * 0.4 + strengthFactor * 0.3 + continuityFactor * 0.3;
  }

  analyzeHierarchicalRelations(components) {
    const relations = new Map();

    // Analyze inter-level relationships
    this.analyzeInterLevelRelations(components, relations);

    // Analyze intra-level relationships
    this.analyzeIntraLevelRelations(components, relations);

    return {
      map: relations,
      statistics: this.calculateRelationStatistics(relations),
      patterns: this.identifyRelationPatterns(relations),
    };
  }

  calculateTopologyMetrics(components) {
    return {
      complexity: this.calculateStructuralComplexity(components),
      coherence: this.calculateStructuralCoherence(components),
      stability: this.calculateStructuralStability(components),
      distributions: this.analyzeFeatureDistributions(components),
    };
  }

  calculateStructuralComplexity(components) {
    const { primary, secondary, junctions, corners } = components;

    const edgeComplexity = this.calculateEdgeComplexity(primary, secondary);
    const junctionComplexity = this.calculateJunctionComplexity(junctions);
    const cornerComplexity = this.calculateCornerComplexity(corners);

    return {
      overall:
        edgeComplexity * 0.5 +
        junctionComplexity * 0.3 +
        cornerComplexity * 0.2,
      components: {
        edges: edgeComplexity,
        junctions: junctionComplexity,
        corners: cornerComplexity,
      },
    };
  }

  // Add these additional analysis functions to EnhancedSignatureGenerator class

  // ====== LBP Rotation Invariance Analysis ======
  analyzeLBPRotationInvariance(histogram) {
    try {
      const rotationInvariantPatterns = new Map();
      let totalPatterns = 0;

      // Group patterns by their rotation invariant equivalents
      for (let pattern = 0; pattern < histogram.length; pattern++) {
        const count = histogram[pattern];
        if (count > 0) {
          const invariantPattern = this.getRotationInvariantPattern(pattern);
          rotationInvariantPatterns.set(
            invariantPattern,
            (rotationInvariantPatterns.get(invariantPattern) || 0) + count
          );
          totalPatterns += count;
        }
      }

      return {
        patterns: rotationInvariantPatterns,
        coverage: rotationInvariantPatterns.size / 36, // 36 unique rotation invariant patterns
        strength: this.calculateRotationInvarianceStrength(
          rotationInvariantPatterns,
          totalPatterns
        ),
        distribution: this.analyzeLBPDistribution(
          rotationInvariantPatterns,
          totalPatterns
        ),
      };
    } catch (error) {
      console.error("Error in analyzeLBPRotationInvariance:", error);
      return {
        patterns: new Map(),
        coverage: 0,
        strength: 0,
        distribution: { entropy: 0, uniformity: 0 },
      };
    }
  }

  getRotationInvariantPattern(pattern) {
    try {
      let minPattern = pattern;
      let binary = pattern.toString(2).padStart(8, "0");

      // Try all rotations and find the minimum value
      for (let i = 0; i < 8; i++) {
        binary = binary.slice(1) + binary[0];
        const rotated = parseInt(binary, 2);
        minPattern = Math.min(minPattern, rotated);
      }

      return minPattern;
    } catch (error) {
      console.error("Error in getRotationInvariantPattern:", error);
      return pattern;
    }
  }

  calculateRotationInvarianceStrength(patterns, total) {
    try {
      if (total === 0) return 0;

      let strength = 0;
      for (const count of patterns.values()) {
        const probability = count / total;
        strength += probability * probability;
      }

      return 1 - Math.sqrt(strength);
    } catch (error) {
      console.error("Error in calculateRotationInvarianceStrength:", error);
      return 0;
    }
  }

  analyzeLBPDistribution(patterns, total) {
    try {
      if (total === 0) return { entropy: 0, uniformity: 0 };

      let entropy = 0;
      let uniformity = 0;

      for (const count of patterns.values()) {
        const probability = count / total;
        if (probability > 0) {
          entropy -= probability * Math.log2(probability);
          uniformity += probability * probability;
        }
      }

      return { entropy, uniformity };
    } catch (error) {
      console.error("Error in analyzeLBPDistribution:", error);
      return { entropy: 0, uniformity: 0 };
    }
  }

  // ====== Histogram Calculation (Fixed Version) ======
  calculateHistogram(values, bins = 256) {
    if (!Array.isArray(values) && !ArrayBuffer.isView(values)) {
      console.error("Invalid input to calculateHistogram:", typeof values);
      return new Float32Array(bins).fill(0);
    }

    // Convert to regular array and find min/max without recursion
    const array = Array.from(values);
    const min = Math.min(...array);
    const max = Math.max(...array);
    const range = max - min || 1; // Prevent division by zero
    const histogram = new Float32Array(bins).fill(0);

    // Calculate histogram iteratively
    for (const value of array) {
      const bin = Math.min(
        bins - 1,
        Math.floor(((value - min) / range) * (bins - 1))
      );
      histogram[bin]++;
    }

    // Normalize
    const total = array.length || 1;
    for (let i = 0; i < bins; i++) {
      histogram[i] /= total;
    }

    return histogram;
  }

  // ====== Distribution Analysis Functions ======
  // ====== Flatten Array Function (Fixed Version) ======
  flattenWaveletCoefficients(coefficients) {
    const result = [];

    // Iterative flattening to prevent stack overflow
    const flatten = (obj) => {
      const stack = [obj];

      while (stack.length > 0) {
        const current = stack.pop();

        if (ArrayBuffer.isView(current)) {
          result.push(...Array.from(current));
        } else if (Array.isArray(current)) {
          result.push(...current);
        } else if (typeof current === "object" && current !== null) {
          stack.push(...Object.values(current));
        }
      }
    };

    flatten(coefficients);
    return result;
  }

  calculateKurtosis(values) {
    if (!values || !Array.isArray(values) || values.length === 0) return 0;

    const mean = this.calculateMean(values);
    const std = this.calculateStd(values);

    if (std === 0) return 0;

    const n = values.length;
    const fourthMoment =
      values.reduce((sum, value) => sum + Math.pow(value - mean, 4), 0) / n;

    return fourthMoment / Math.pow(std, 4) - 3;
  }

  // ====== Radon Distinctiveness Analysis ======
  calculateRadonDistinctiveness(transform) {
    const { features, statistics } = this.extractRadonFeatures(transform);

    return {
      featureDistinctiveness: this.calculateFeatureDistinctiveness(features),
      statisticalDistinctiveness:
        this.calculateStatisticalDistinctiveness(statistics),
      overallScore: this.combineRadonDistinctiveness(features, statistics),
    };
  }

  extractRadonFeatures(transform) {
    const features = {
      peaks: this.findRadonPeaks(transform),
      valleys: this.findRadonValleys(transform),
      gradients: this.calculateRadonGradients(transform),
    };

    const statistics = {
      mean: this.calculateMean(transform),
      std: this.calculateStd(transform),
      skewness: this.calculateSkewness(transform),
      kurtosis: this.calculateKurtosis(transform),
    };

    return { features, statistics };
  }

  findRadonPeaks(transform) {
    const peaks = [];
    const threshold =
      this.calculateMean(transform) + this.calculateStd(transform);

    for (let i = 1; i < transform.length - 1; i++) {
      if (
        transform[i] > threshold &&
        transform[i] > transform[i - 1] &&
        transform[i] > transform[i + 1]
      ) {
        peaks.push({
          position: i,
          value: transform[i],
          prominence: Math.min(
            transform[i] - transform[i - 1],
            transform[i] - transform[i + 1]
          ),
        });
      }
    }

    return peaks;
  }

  findRadonValleys(transform) {
    const valleys = [];
    const threshold =
      this.calculateMean(transform) - this.calculateStd(transform);

    for (let i = 1; i < transform.length - 1; i++) {
      if (
        transform[i] < threshold &&
        transform[i] < transform[i - 1] &&
        transform[i] < transform[i + 1]
      ) {
        valleys.push({
          position: i,
          value: transform[i],
          depth: Math.min(
            transform[i - 1] - transform[i],
            transform[i + 1] - transform[i]
          ),
        });
      }
    }

    return valleys;
  }

  calculateRadonGradients(transform) {
    const gradients = new Float32Array(transform.length - 1);
    for (let i = 0; i < transform.length - 1; i++) {
      gradients[i] = transform[i + 1] - transform[i];
    }
    return gradients;
  }

  calculateValleyDistinctiveness(valleys) {
    if (!valleys || !Array.isArray(valleys)) {
      return 0;
    }

    // Sort valleys by depth
    const sortedValleys = [...valleys].sort((a, b) => b.depth - a.depth);

    if (sortedValleys.length === 0) return 0;

    // Calculate relative depths and spacings
    let distinctiveness = 0;
    let totalWeight = 0;

    for (let i = 0; i < sortedValleys.length - 1; i++) {
      const currentValley = sortedValleys[i];
      const nextValley = sortedValleys[i + 1];

      // Calculate depth ratio
      const depthRatio = nextValley.depth / currentValley.depth;

      // Calculate spacing
      const spacing = Math.abs(nextValley.position - currentValley.position);

      // Weight based on valley depth
      const weight = currentValley.depth;

      // Combine metrics
      distinctiveness +=
        weight * (depthRatio * 0.5 + Math.min(spacing / 10, 1) * 0.5);
      totalWeight += weight;
    }

    return totalWeight > 0 ? distinctiveness / totalWeight : 0;
  }

  calculateFeatureDistinctiveness(features) {
    const peakScore = this.calculatePeakDistinctiveness(features.peaks);
    const valleyScore = this.calculateValleyDistinctiveness(features.valleys);
    const gradientScore = this.calculateGradientDistinctiveness(
      features.gradients
    );

    return peakScore * 0.4 + valleyScore * 0.3 + gradientScore * 0.3;
  }

  calculateStatisticalDistinctiveness(statistics) {
    const normalizedStats = {
      skewness:
        Math.abs(statistics.skewness) / (1 + Math.abs(statistics.skewness)),
      kurtosis:
        Math.abs(statistics.kurtosis) / (1 + Math.abs(statistics.kurtosis)),
      variation: statistics.std / (Math.abs(statistics.mean) + 1e-6),
    };

    return (
      normalizedStats.skewness * 0.3 +
      normalizedStats.kurtosis * 0.3 +
      normalizedStats.variation * 0.4
    );
  }

  assessLBPRotationInvariance(rotationData) {
    try {
      if (
        !rotationData ||
        !rotationData.patterns ||
        typeof rotationData.patterns.get !== "function"
      ) {
        console.warn("Invalid rotation data in assessLBPRotationInvariance");
        return 0;
      }

      const patterns = rotationData.patterns;
      let totalPatterns = 0;
      let rotationInvariantCount = 0;

      // Calculate total patterns and rotation invariant patterns
      for (const [pattern, count] of patterns.entries()) {
        const rotInvPattern = this.getRotationInvariantPattern(pattern);
        if (rotInvPattern === pattern) {
          rotationInvariantCount += count;
        }
        totalPatterns += count;
      }

      if (totalPatterns === 0) return 0;

      // Calculate rotation invariance score
      const invarianceRatio = rotationInvariantCount / totalPatterns;
      const patternDiversity = patterns.size / 256; // Normalize by max possible patterns

      return invarianceRatio * 0.7 + patternDiversity * 0.3;
    } catch (error) {
      console.error("Error in assessLBPRotationInvariance:", error);
      return 0;
    }
  }

  // ====== LBP Quality Assessment ======
  assessLBPQuality(features) {
    if (!features || !features.histogram) return 0;

    const uniformity = this.calculateLBPUniformity(features.histogram);
    const entropy = this.calculateLBPEntropy(features.histogram);
    const variance = this.calculateLBPVariance(features.histogram);

    return uniformity * 0.4 + entropy * 0.3 + variance * 0.3;
  }

  assessLBPStability(patterns) {
    try {
      // Ensure patterns is iterable
      if (!patterns || typeof patterns.values !== "function") {
        console.warn("Invalid patterns object in assessLBPStability");
        return 0;
      }

      const patternCounts = Array.from(patterns.values());
      const total = patternCounts.reduce((sum, count) => sum + count, 0);

      if (total === 0) return 0;

      // Calculate normalized entropy
      const probabilities = patternCounts.map((count) => count / total);
      const entropy = -probabilities.reduce(
        (sum, p) => sum + (p > 0 ? p * Math.log2(p) : 0),
        0
      );

      // Normalize entropy to [0,1]
      const maxEntropy = Math.log2(patterns.size || 1);
      return maxEntropy === 0 ? 0 : entropy / maxEntropy;
    } catch (error) {
      console.error("Error in assessLBPStability:", error);
      return 0;
    }
  }

  assessLBPDistinctiveness(patterns) {
    const counts = Array.from(patterns.values());
    const total = counts.reduce((sum, count) => sum + count, 0);

    if (total === 0) return 0;

    // Calculate Gini coefficient for pattern distribution
    counts.sort((a, b) => a - b);
    let sumCumulative = 0;
    let sumScores = counts.reduce((sum, count, i) => {
      sumCumulative += count;
      return sum + sumCumulative * count;
    }, 0);

    const n = counts.length;
    const denominator = total * sumCumulative;
    return denominator > 0 ? (n + 1 - (2 * sumScores) / denominator) / n : 0;
  }

  calculateLBPConfidence(metrics) {
    try {
      // Calculate confidence based on metric consistency
      const values = Object.values(metrics);
      if (values.length === 0) return 0;

      const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
      if (mean === 0) return 0;

      const std = Math.sqrt(
        values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) /
          values.length
      );

      // Higher confidence when metrics are consistent
      return Math.exp(-std / mean);
    } catch (error) {
      console.error("Error in calculateLBPConfidence:", error);
      return 0;
    }
  }

  // ====== Hash Stability Analysis ======
  calculateHashStability(hash) {
    return {
      bitStability: this.assessBitStability(hash),
      noiseStability: this.assessNoiseStability(hash),
      geometricStability: this.assessGeometricStability(hash),
      statisticalStability: this.assessStatisticalStability(hash),
    };
  }

  simulateScaling(hash, scale) {
    const size = Math.sqrt(hash.length * 8);
    const scaled = new Uint8Array(hash.length);
    const center = size / 2;

    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const dx = (x - center) / scale + center;
        const dy = (y - center) / scale + center;

        if (dx >= 0 && dx < size && dy >= 0 && dy < size) {
          const oldBit = this.getBit(
            hash,
            Math.round(dx),
            Math.round(dy),
            size
          );
          this.setBit(scaled, x, y, size, oldBit);
        }
      }
    }

    return scaled;
  }

  // Add after simulateScaling
  calculateWaveletCompressionRatio(coefficients) {
    if (!coefficients || !coefficients.coefficients) return 0;

    const allCoeffs = this.flattenWaveletCoefficients(
      coefficients.coefficients
    );
    if (allCoeffs.length === 0) return 0;

    // Count significant coefficients (above threshold)
    const threshold = this.calculateWaveletThreshold(allCoeffs);
    const significantCoeffs = allCoeffs.filter((c) => Math.abs(c) > threshold);

    return 1 - significantCoeffs.length / allCoeffs.length;
  }

  // Add helper method for wavelet compression calculation
  calculateWaveletThreshold(coefficients) {
    if (!coefficients || coefficients.length === 0) return 0;

    const sorted = Array.from(coefficients).sort(
      (a, b) => Math.abs(b) - Math.abs(a)
    );
    const medianIndex = Math.floor(sorted.length * 0.5);
    return Math.abs(sorted[medianIndex]) * 0.1;
  }

  assessBitStability(hash) {
    const bitTransitions = this.countBitTransitions(hash);
    const maxTransitions = hash.length * 8 - 1;
    return 1 - bitTransitions / maxTransitions;
  }

  countBitTransitions(hash) {
    let transitions = 0;
    for (let i = 0; i < hash.length; i++) {
      let byte = hash[i];
      for (let bit = 1; bit < 8; bit++) {
        if (((byte >> bit) & 1) !== ((byte >> (bit - 1)) & 1)) {
          transitions++;
        }
      }
      if (i < hash.length - 1) {
        if (((byte >> 7) & 1) !== (hash[i + 1] & 1)) {
          transitions++;
        }
      }
    }
    return transitions;
  }

  assessNoiseStability(hash) {
    const trials = 10;
    let stability = 0;

    for (let i = 0; i < trials; i++) {
      const noisyHash = this.addNoiseToHash(hash, 0.1);
      stability += this.calculateHashSimilarity(hash, noisyHash);
    }

    return stability / trials;
  }

  addNoiseToHash(hash, noiseLevel) {
    const noisyHash = new Uint8Array(hash.length);
    for (let i = 0; i < hash.length; i++) {
      noisyHash[i] = hash[i];
      for (let bit = 0; bit < 8; bit++) {
        if (Math.random() < noiseLevel) {
          noisyHash[i] ^= 1 << bit;
        }
      }
    }
    return noisyHash;
  }

  calculateHashSimilarity(hash1, hash2) {
    if (!hash1 || !hash2 || hash1.length !== hash2.length) {
      return 0;
    }

    let matchingBits = 0;
    const totalBits = hash1.length * 8;

    for (let i = 0; i < hash1.length; i++) {
      const xor = hash1[i] ^ hash2[i];
      matchingBits += 8 - this.popCount(xor);
    }

    return matchingBits / totalBits;
  }

  async extractColorFeaturesWithQuality(buffer) {
    const startTime = process.hrtime.bigint();
    const features = await this.extractColorFeatures(buffer);
    const duration = Number(process.hrtime.bigint() - startTime) / 1_000_000;

    logger.info("Color Feature Analysis:", {
      dominantColors: features.dominant?.length || 0,
      distributionEntropy: features.distribution?.entropy || 0,
      processingTime: duration,
      quality: {
        overall: features.quality?.overall || 0,
        components: {
          dominant: features.quality?.components?.dominant || 0,
          distribution: features.quality?.components?.distribution || 0,
          harmony: features.quality?.components?.harmony || 0,
        },
      },
    });

    return features;
  }

  async extractEdgeFeaturesWithQuality(buffer) {
    const startTime = process.hrtime.bigint();
    const features = await this.extractEdgeFeatures(buffer);
    const duration = Number(process.hrtime.bigint() - startTime) / 1_000_000;

    logger.info("Edge Feature Analysis:", {
      edgeCount: features.edges?.length || 0,
      strength: features.features?.strength || 0,
      continuity: features.features?.continuity?.score || 0,
      processingTime: duration,
      quality: {
        overall: features.quality?.overall || 0,
        components: {
          strength: features.quality?.components?.strength || 0,
          continuity: features.quality?.components?.continuity || 0,
          distribution: features.quality?.components?.distribution || 0,
          junctions: features.quality?.components?.junctions || 0,
        },
      },
    });

    return features;
  }

  async extractTextureFeaturesWithQuality(buffer) {
    const startTime = process.hrtime.bigint();
    const features = await this.extractTextureFeatures(buffer);
    const duration = Number(process.hrtime.bigint() - startTime) / 1_000_000;

    logger.info("Texture Feature Analysis:", {
      glcmFeatures: features.glcm ? "Present" : "Missing",
      lbpFeatures: features.lbp ? "Present" : "Missing",
      gaborFeatures: features.gabor ? "Present" : "Missing",
      processingTime: duration,
      quality: {
        overall: features.quality?.overall || 0,
        components: {
          glcm: features.quality?.components?.glcm || 0,
          lbp: features.quality?.components?.lbp || 0,
          gabor: features.quality?.components?.gabor || 0,
        },
      },
    });

    return features;
  }

  async generatePerceptualHashesWithQuality(buffer) {
    const startTime = process.hrtime.bigint();
    const hashes = await this.generatePerceptualHashes(buffer);
    const duration = Number(process.hrtime.bigint() - startTime) / 1_000_000;

    logger.info("Hash Generation Analysis:", {
      dctHash: hashes.dct ? "Generated" : "Failed",
      waveletHash: hashes.wavelet ? "Generated" : "Failed",
      radonHash: hashes.radon ? "Generated" : "Failed",
      processingTime: duration,
      quality: {
        dct: {
          overall: hashes.dct?.quality?.overall || 0,
          components: {
            robustness: hashes.dct?.quality?.robustness || 0,
            distinctiveness: hashes.dct?.quality?.distinctiveness || 0,
            stability: hashes.dct?.quality?.stability || 0,
          },
        },
        wavelet: {
          overall: hashes.wavelet?.quality?.overall || 0,
          components: hashes.wavelet?.quality?.components || {},
        },
        radon: {
          overall: hashes.radon?.quality?.overall || 0,
          components: hashes.radon?.quality?.components || {},
        },
      },
    });

    return hashes;
  }

  async generateSpatialVerificationWithQuality(buffer) {
    const startTime = process.hrtime.bigint();
    const features = await this.generateSpatialVerification(buffer);
    const duration = Number(process.hrtime.bigint() - startTime) / 1_000_000;

    logger.info("Spatial Verification Analysis:", {
      gridSize: features.grid?.length || 0,
      regionCount: features.regions?.length || 0,
      relationshipCount: features.relationships?.length || 0,
      processingTime: duration,
      quality: {
        overall: features.quality?.overall || 0,
        components: {
          grid: features.quality?.components?.grid || 0,
          regions: features.quality?.components?.regions || 0,
          relationships: features.quality?.components?.relationships || 0,
        },
      },
    });

    return features;
  }

  logFeatureQualities(qualities) {
    logger.info("Feature Quality Analysis:", {
      color: {
        overall: qualities.color?.overall || 0,
        components: {
          dominantColors: qualities.color?.components?.dominant || 0,
          distribution: qualities.color?.components?.distribution || 0,
          harmony: qualities.color?.components?.harmony || 0,
        },
      },
      edge: {
        overall: qualities.edge?.overall || 0,
        components: {
          strength: qualities.edge?.components?.strength || 0,
          continuity: qualities.edge?.components?.continuity || 0,
          distribution: qualities.edge?.components?.distribution || 0,
          junctions: qualities.edge?.components?.junctions || 0,
        },
      },
      texture: {
        overall: qualities.texture?.overall || 0,
        components: {
          glcm: qualities.texture?.components?.glcm || 0,
          lbp: qualities.texture?.components?.lbp || 0,
          gabor: qualities.texture?.components?.gabor || 0,
        },
      },
      hash: {
        overall: qualities.hash?.overall || 0,
        components: {
          robustness: qualities.hash?.components?.robustness || 0,
          distinctiveness: qualities.hash?.components?.distinctiveness || 0,
          stability: qualities.hash?.components?.stability || 0,
        },
      },
    });
  }

  assessGeometricStability(hash) {
    const rotationStability = this.assessRotationStability(hash);
    const scaleStability = this.assessScaleStability(hash);
    const translationStability = this.assessTranslationStability(hash);

    return (
      rotationStability * 0.4 +
      scaleStability * 0.3 +
      translationStability * 0.3
    );
  }

  assessStatisticalStability(hash) {
    if (!hash || !Buffer.isBuffer(hash)) {
      return 0;
    }

    try {
      const bitCounts = new Array(8).fill(0);
      for (const byte of hash) {
        for (let bit = 0; bit < 8; bit++) {
          if ((byte >> bit) & 1) bitCounts[bit]++;
        }
      }

      const mean = this.calculateMean(bitCounts);
      const std = this.calculateStd(bitCounts);
      return mean === 0 ? 0 : 1 - std / mean;
    } catch (error) {
      console.error("Statistical stability assessment failed:", error);
      return 0;
    }
  }

  // ====== Interscale Statistics ======
  calculateInterscaleStatistics(coefficients) {
    return {
      correlations: this.calculateInterscaleCorrelations(coefficients),
      ratios: this.calculateInterscaleRatios(coefficients),
      coherence: this.calculateInterscaleCoherence(coefficients),
    };
  }

  calculateInterscaleCorrelations(coefficients) {
    const bands = ["LL", "LH", "HL", "HH"];
    const correlations = {};

    for (let i = 0; i < bands.length; i++) {
      for (let j = i + 1; j < bands.length; j++) {
        const correlation = this.calculateBandCorrelation(
          coefficients[bands[i]],
          coefficients[bands[j]]
        );
        correlations[`${bands[i]}_${bands[j]}`] = correlation;
      }
    }

    return correlations;
  }

  calculateBandCorrelation(band1, band2) {
    const meanA = this.calculateMean(Array.from(band1));
    const meanB = this.calculateMean(Array.from(band2));
    let numerator = 0;
    let denomA = 0;
    let denomB = 0;

    for (let i = 0; i < band1.length; i++) {
      const diffA = band1[i] - meanA;
      const diffB = band2[i] - meanB;
      numerator += diffA * diffB;
      denomA += diffA * diffA;
      denomB += diffB * diffB;
    }

    return numerator / Math.sqrt(denomA * denomB);
  }

  calculateInterscaleRatios(coefficients) {
    const bands = ["LL", "LH", "HL", "HH"];
    const ratios = {};

    for (let i = 0; i < bands.length; i++) {
      for (let j = i + 1; j < bands.length; j++) {
        const energyA = this.calculateBandEnergy(coefficients[bands[i]]);
        const energyB = this.calculateBandEnergy(coefficients[bands[j]]);
        ratios[`${bands[i]}_${bands[j]}`] = energyA / (energyB + 1e-6);
      }
    }

    return ratios;
  }

  calculateInterscaleCoherence(coefficients) {
    const bands = ["LL", "LH", "HL", "HH"];
    let totalCoherence = 0;

    for (let i = 0; i < bands.length - 1; i++) {
      const coherence = this.calculateBandCoherence(
        coefficients[bands[i]],
        coefficients[bands[i + 1]]
      );
      totalCoherence += coherence;
    }

    return totalCoherence / (bands.length - 1);
  }

  calculateBandCoherence(band1, band2) {
    const magnitudes1 = Array.from(band1).map(Math.abs);
    const magnitudes2 = Array.from(band2).map(Math.abs);

    const mean1 = this.calculateMean(magnitudes1);
    const mean2 = this.calculateMean(magnitudes2);

    let coherence = 0;
    for (let i = 0; i < magnitudes1.length; i++) {
      coherence += Math.min(magnitudes1[i] / mean1, magnitudes2[i] / mean2);
    }

    return coherence / magnitudes1.length;
  }

  // ====== Peak Distinctiveness Analysis ======
  calculatePeakDistinctiveness(peaks) {
    if (!peaks || peaks.length === 0) return 0;

    const prominenceScores = peaks.map((peak) =>
      this.calculatePeakProminence(peak)
    );
    const spacingScores = this.calculatePeakSpacing(peaks);
    const stabilityScores = this.calculatePeakStability(peaks);

    return {
      prominence: this.calculateMean(prominenceScores),
      spacing: spacingScores,
      stability: stabilityScores,
      overall:
        this.calculateMean(prominenceScores) * 0.4 +
        spacingScores * 0.3 +
        stabilityScores * 0.3,
    };
  }

  calculatePeakProminence(peak) {
    return peak.prominence / (peak.value + 1e-6);
  }

  calculatePeakSpacing(peaks) {
    if (peaks.length < 2) return 0;

    const spacings = [];
    for (let i = 1; i < peaks.length; i++) {
      spacings.push(Math.abs(peaks[i].position - peaks[i - 1].position));
    }

    const meanSpacing = this.calculateMean(spacings);
    const stdSpacing = this.calculateStd(spacings);

    return 1 / (1 + stdSpacing / meanSpacing);
  }

  calculatePeakStability(peaks) {
    if (peaks.length === 0) return 0;

    const values = peaks.map((p) => p.value);
    const mean = this.calculateMean(values);
    const std = this.calculateStd(values);

    return 1 / (1 + std / mean);
  }

  combineRadonDistinctiveness(features, statistics) {
    const featureScore = this.calculateFeatureDistinctiveness(features);
    const statScore = this.calculateStatisticalDistinctiveness(statistics);

    // Weight feature distinctiveness more heavily
    return featureScore * 0.6 + statScore * 0.4;
  }

  getDefaultLBPFeatures() {
    return {
      histogram: new Float32Array(256).fill(1 / 256),
      patterns: {
        distribution: new Map(),
        uniformity: 0,
        dominantPatterns: [],
      },
      uniformity: 0,
      rotation: {
        invariance: 0,
        patterns: new Map(),
      },
      quality: {
        overall: 0,
        components: {
          uniformity: 0,
          stability: 0,
          distinctiveness: 0,
        },
      },
    };
  }

  getDefaultWaveletFeatures() {
    return {
      coefficients: {
        LL: new Float32Array(0),
        LH: new Float32Array(0),
        HL: new Float32Array(0),
        HH: new Float32Array(0),
      },
      statistics: {
        mean: 0,
        std: 0,
        energy: 0,
        entropy: 0,
      },
      energy: {
        LL: 0,
        LH: 0,
        HL: 0,
        HH: 0,
      },
      entropy: {
        LL: 0,
        LH: 0,
        HL: 0,
        HH: 0,
      },
      quality: {
        overall: 0,
        components: {
          energy: 0,
          entropy: 0,
          persistence: 0,
        },
      },
    };
  }

  computeLBP(data, width, height) {
    try {
      const lbpImage = new Uint8Array(width * height);
      const radius = Math.max(
        1,
        Math.min(
          this.config.TEXTURE.LBP_RADIUS,
          Math.floor(Math.min(width, height) / 4)
        )
      );
      const neighbors = 8;

      // Adjust loop bounds to avoid edge issues
      for (let y = radius; y < height - radius; y++) {
        for (let x = radius; x < width - radius; x++) {
          let pattern = 0;
          const centerPixel = data[y * width + x];

          for (let n = 0; n < neighbors; n++) {
            const theta = (2 * Math.PI * n) / neighbors;
            const xn = Math.round(x + radius * Math.cos(theta));
            const yn = Math.round(y + radius * Math.sin(theta));

            if (xn >= 0 && xn < width && yn >= 0 && yn < height) {
              const neighbor = data[yn * width + xn];
              if (neighbor >= centerPixel) {
                pattern |= 1 << n;
              }
            }
          }

          lbpImage[y * width + x] = pattern;
        }
      }

      return lbpImage;
    } catch (error) {
      logger.error("LBP computation failed:", error);
      return new Uint8Array(width * height);
    }
  }

  computeLBPHistogram(lbpImage) {
    const histogram = new Float32Array(256).fill(0);
    for (const pattern of lbpImage) {
      histogram[pattern]++;
    }

    // Normalize histogram
    const sum = histogram.reduce((a, b) => a + b, 0);
    if (sum > 0) {
      for (let i = 0; i < histogram.length; i++) {
        histogram[i] /= sum;
      }
    }

    return histogram;
  }

  applyHaarWavelet(data, size) {
    const coefficients = new Float32Array(data);

    // Apply 1D Haar wavelet transform to rows and columns
    for (let i = 0; i < size; i++) {
      this.haarTransform1D(coefficients, i * size, size);
      this.haarTransform1D(coefficients, i, size, size);
    }

    return coefficients;
  }

  haarTransform1D(data, offset, length, stride = 1) {
    if (length === 1) return;

    const halfLength = length >> 1;
    const temp = new Float32Array(length);

    for (let i = 0; i < halfLength; i++) {
      const even = data[offset + i * 2 * stride];
      const odd = data[offset + (i * 2 + 1) * stride];

      temp[i] = (even + odd) / Math.SQRT2;
      temp[i + halfLength] = (even - odd) / Math.SQRT2;
    }

    for (let i = 0; i < length; i++) {
      data[offset + i * stride] = temp[i];
    }
  }

  calculateWaveletMedian(coefficients) {
    const sorted = coefficients.slice().sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0
      ? (sorted[mid - 1] + sorted[mid]) / 2
      : sorted[mid];
  }

  calculateHashRobustness(hash) {
    const bitDistribution = this.analyzeBitDistribution(hash);
    const patternDiversity = this.analyzePatternDiversity(hash);
    const noiseResistance = this.calculateNoiseResistance(hash);

    return {
      overall:
        bitDistribution * 0.4 + patternDiversity * 0.3 + noiseResistance * 0.3,
      components: {
        distribution: bitDistribution,
        diversity: patternDiversity,
        resistance: noiseResistance,
      },
    };
  }

  analyzeBitDistribution(hash) {
    let ones = 0;
    const totalBits = hash.length * 8;

    for (let i = 0; i < hash.length; i++) {
      ones += this.popCount(hash[i]);
    }

    const ratio = ones / totalBits;
    return 1 - Math.abs(0.5 - ratio) * 2;
  }

  calculateNoiseResistance(hash) {
    const samples = 10;
    let totalResistance = 0;

    for (let i = 0; i < samples; i++) {
      const noisyHash = this.addRandomNoise(hash, 0.1);
      const similarity = this.calculateHashSimilarity(hash, noisyHash);
      totalResistance += similarity;
    }

    return totalResistance / samples;
  }

  addRandomNoise(hash, noiseLevel) {
    const noisyHash = new Uint8Array(hash.length);

    for (let i = 0; i < hash.length; i++) {
      noisyHash[i] = hash[i];
      for (let bit = 0; bit < 8; bit++) {
        if (Math.random() < noiseLevel) {
          noisyHash[i] ^= 1 << bit;
        }
      }
    }

    return noisyHash;
  }

  calculateHashSimilarity(hash1, hash2) {
    let diffBits = 0;
    const totalBits = hash1.length * 8;

    for (let i = 0; i < hash1.length; i++) {
      diffBits += this.popCount(hash1[i] ^ hash2[i]);
    }

    return 1 - diffBits / totalBits;
  }

  // Add these texture analysis implementations to EnhancedSignatureGenerator class

  async computeGLCM(data, width, height, distance, angle) {
    const glcm = Array(256)
      .fill()
      .map(() => Array(256).fill(0));
    const angleRad = (angle * Math.PI) / 180;
    const dx = Math.round(Math.cos(angleRad) * distance);
    const dy = Math.round(Math.sin(angleRad) * distance);

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const nx = x + dx;
        const ny = y + dy;

        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
          const i = data[y * width + x];
          const j = data[ny * width + nx];
          glcm[i][j]++;
        }
      }
    }

    return glcm;
  }

  async normalizeGLCM(glcm) {
    const normalized = Array(256)
      .fill()
      .map(() => Array(256).fill(0));
    let sum = 0;

    // Calculate sum
    for (let i = 0; i < 256; i++) {
      for (let j = 0; j < 256; j++) {
        sum += glcm[i][j];
      }
    }

    // Normalize
    if (sum > 0) {
      for (let i = 0; i < 256; i++) {
        for (let j = 0; j < 256; j++) {
          normalized[i][j] = glcm[i][j] / sum;
        }
      }
    }

    return normalized;
  }

  async calculateGLCMFeatures(normalizedGLCM) {
    const features = {
      contrast: 0,
      dissimilarity: 0,
      homogeneity: 0,
      energy: 0,
      correlation: 0,
      entropy: 0,
    };

    let meanI = 0,
      meanJ = 0;
    let stdI = 0,
      stdJ = 0;

    // Calculate means
    for (let i = 0; i < 256; i++) {
      for (let j = 0; j < 256; j++) {
        meanI += i * normalizedGLCM[i][j];
        meanJ += j * normalizedGLCM[i][j];
      }
    }

    // Calculate standard deviations
    for (let i = 0; i < 256; i++) {
      for (let j = 0; j < 256; j++) {
        stdI += Math.pow(i - meanI, 2) * normalizedGLCM[i][j];
        stdJ += Math.pow(j - meanJ, 2) * normalizedGLCM[i][j];
      }
    }

    stdI = Math.sqrt(stdI);
    stdJ = Math.sqrt(stdJ);

    // Calculate features
    for (let i = 0; i < 256; i++) {
      for (let j = 0; j < 256; j++) {
        const p = normalizedGLCM[i][j];
        if (p > 0) {
          features.contrast += Math.pow(i - j, 2) * p;
          features.dissimilarity += Math.abs(i - j) * p;
          features.homogeneity += p / (1 + Math.pow(i - j, 2));
          features.energy += p * p;
          features.entropy -= p * Math.log(p);

          if (stdI > 0 && stdJ > 0) {
            features.correlation +=
              ((i - meanI) * (j - meanJ) * p) / (stdI * stdJ);
          }
        }
      }
    }

    return features;
  }

  async calculateGLCMStatistics(features) {
    const statistics = {};
    const featureNames = [
      "contrast",
      "dissimilarity",
      "homogeneity",
      "energy",
      "correlation",
      "entropy",
    ];

    for (const feature of featureNames) {
      const values = [];
      for (const distance in features) {
        for (const angle in features[distance]) {
          values.push(features[distance][angle][feature]);
        }
      }

      statistics[feature] = {
        mean: this.calculateArrayMean(values),
        std: this.calculateArrayStd(values),
        min: Math.min(...values),
        max: Math.max(...values),
        range: Math.max(...values) - Math.min(...values),
      };
    }

    return statistics;
  }

  calculateArrayMean(values) {
    if (!values || values.length === 0) return 0;
    return values.reduce((sum, val) => sum + val, 0) / values.length;
  }

  calculateArrayStd(values) {
    if (!values || values.length === 0) return 0;
    const mean = this.calculateArrayMean(values);
    const squaredDiffs = values.map((val) => Math.pow(val - mean, 2));
    const variance = this.calculateArrayMean(squaredDiffs);
    return Math.sqrt(variance);
  }

  assessGLCMQuality(features) {
    if (!features || !features.features) return 0;

    const metrics = {
      contrast: features.features.contrast || 0,
      correlation: Math.abs(features.features.correlation || 0),
      energy: features.features.energy || 0,
      homogeneity: features.features.homogeneity || 0,
    };

    const weights = {
      contrast: 0.3,
      correlation: 0.2,
      energy: 0.25,
      homogeneity: 0.25,
    };

    return Object.entries(metrics).reduce(
      (quality, [metric, value]) => quality + weights[metric] * value,
      0
    );
  }

  assessStatisticalValidity(statistics) {
    // Check if features are within expected ranges
    const validRanges = {
      contrast: [0, Infinity],
      dissimilarity: [0, Infinity],
      homogeneity: [0, 1],
      energy: [0, 1],
      correlation: [-1, 1],
      entropy: [0, Infinity],
    };

    let validCount = 0;
    const totalChecks = Object.keys(validRanges).length;

    for (const [feature, range] of Object.entries(validRanges)) {
      const stats = statistics[feature];
      if (
        stats &&
        stats.min >= range[0] &&
        (range[1] === Infinity || stats.max <= range[1])
      ) {
        validCount++;
      }
    }

    return validCount / totalChecks;
  }

  assessFeatureConsistency(statistics) {
    // Check consistency across different angles and distances
    let consistencyScore = 0;
    const features = Object.keys(statistics);

    for (const feature of features) {
      const stats = statistics[feature];
      const relativeRange = stats.range / (stats.max - stats.min + 1e-6);
      consistencyScore += 1 - Math.min(1, relativeRange);
    }

    return consistencyScore / features.length;
  }

  assessFeatureReliability(features) {
    // Assess reliability based on feature stability
    let reliabilityScore = 0;
    let count = 0;

    for (const distance in features) {
      for (const angle in features[distance]) {
        const angleFeatures = features[distance][angle];

        // Check for reasonable feature values
        if (
          angleFeatures.energy >= 0 &&
          angleFeatures.energy <= 1 &&
          angleFeatures.correlation >= -1 &&
          angleFeatures.correlation <= 1 &&
          angleFeatures.homogeneity >= 0 &&
          angleFeatures.homogeneity <= 1
        ) {
          reliabilityScore++;
        }
        count++;
      }
    }

    return count > 0 ? reliabilityScore / count : 0;
  }

  getDefaultGLCMFeatures() {
    return {
      features: {},
      statistics: {
        contrast: { mean: 0, std: 0, min: 0, max: 0, range: 0 },
        dissimilarity: { mean: 0, std: 0, min: 0, max: 0, range: 0 },
        homogeneity: { mean: 0, std: 0, min: 0, max: 0, range: 0 },
        energy: { mean: 0, std: 0, min: 0, max: 0, range: 0 },
        correlation: { mean: 0, std: 0, min: 0, max: 0, range: 0 },
        entropy: { mean: 0, std: 0, min: 0, max: 0, range: 0 },
      },
      quality: {
        overall: 0,
        components: {
          validity: 0,
          consistency: 0,
          reliability: 0,
        },
      },
    };
  }

  // Helper function to validate texture analysis input
  validateTextureInput(data, width, height) {
    if (!data || !width || !height) {
      throw new Error("Invalid input parameters for texture analysis");
    }
    if (data.length !== width * height) {
      throw new Error("Data dimensions do not match provided width and height");
    }
    return true;
  }

  calculateTextureComplexity(texture) {
    try {
      if (!texture) return 0;

      const complexityFactors = {
        lbp: this.calculateLBPComplexity(texture.lbp || []),
        glcm: this.calculateGLCMComplexity(texture.glcm || []),
        gabor: this.calculateGaborComplexity(texture.gabor || []),
      };

      return (
        complexityFactors.lbp * 0.4 +
        complexityFactors.glcm * 0.3 +
        complexityFactors.gabor * 0.3
      );
    } catch (error) {
      console.error("Texture complexity calculation failed:", error);
      return 0;
    }
  }

  analyzeTexturePatterns(cellTexture) {
    try {
      if (!cellTexture) {
        return {
          uniformity: 0,
          complexity: 0,
          regularity: 0,
          orientation: 0,
        };
      }

      return {
        uniformity: this.calculateTextureUniformity(cellTexture),
        complexity: this.calculateTextureComplexity(cellTexture),
        regularity: this.calculateTextureRegularity(cellTexture),
        orientation: this.calculateTextureOrientation(cellTexture),
      };
    } catch (error) {
      console.error("Texture pattern analysis failed:", error);
      return {
        uniformity: 0,
        complexity: 0,
        regularity: 0,
        orientation: 0,
      };
    }
  }

  calculateTextureUniformity(texture) {
    try {
      if (!texture || !texture.lbp || !texture.lbp.length) return 0;

      const histogram = texture.lbp.reduce((acc, pattern) => {
        if (pattern.histogram) {
          pattern.histogram.forEach((value, index) => {
            acc[index] = (acc[index] || 0) + value;
          });
        }
        return acc;
      }, []);

      if (!histogram.length) return 0;

      const sum = histogram.reduce((a, b) => a + b, 0);
      if (sum === 0) return 0;

      return histogram.reduce((uniformity, count) => {
        const p = count / sum;
        return uniformity + p * p;
      }, 0);
    } catch (error) {
      console.error("Texture uniformity calculation failed:", error);
      return 0;
    }
  }

  calculateLBPComplexity(lbpFeatures) {
    try {
      if (!lbpFeatures || !Array.isArray(lbpFeatures)) return 0;

      const patterns = lbpFeatures.reduce((acc, feat) => {
        if (feat.histogram) acc.push(...feat.histogram);
        return acc;
      }, []);

      if (patterns.length === 0) return 0;

      // Calculate pattern diversity
      const uniquePatterns = new Set(patterns).size;
      const maxPossiblePatterns = 256; // For 8-bit LBP
      const diversity = uniquePatterns / maxPossiblePatterns;

      // Calculate pattern transitions
      let transitions = 0;
      for (let i = 0; i < patterns.length - 1; i++) {
        if (patterns[i] !== patterns[i + 1]) transitions++;
      }
      const transitionRate = transitions / (patterns.length - 1 || 1);

      return diversity * 0.6 + transitionRate * 0.4;
    } catch (error) {
      console.error("LBP complexity calculation failed:", error);
      return 0;
    }
  }

  calculateGLCMComplexity(glcmFeatures) {
    try {
      if (!glcmFeatures || !Array.isArray(glcmFeatures)) return 0;

      const complexityFactors = glcmFeatures.map((feature) => {
        if (!feature) return 0;

        const contrast = feature.contrast || 0;
        const entropy = feature.entropy || 0;
        const correlation = Math.abs(feature.correlation || 0);

        return contrast * 0.4 + entropy * 0.4 + correlation * 0.2;
      });

      return (
        complexityFactors.reduce((sum, factor) => sum + factor, 0) /
        Math.max(1, complexityFactors.length)
      );
    } catch (error) {
      console.error("GLCM complexity calculation failed:", error);
      return 0;
    }
  }

  calculateGaborComplexity(gaborFeatures) {
    try {
      if (!gaborFeatures || !Array.isArray(gaborFeatures)) return 0;

      const validResponses = gaborFeatures.filter(
        (f) => f && typeof f.magnitude === "number"
      );
      if (validResponses.length === 0) return 0;

      const magnitudes = validResponses.map((f) => f.magnitude);
      const mean = this.calculateMean(magnitudes);
      const std = this.calculateStd(magnitudes);
      const variation = mean > 0 ? std / mean : 0;

      return Math.min(1, variation);
    } catch (error) {
      console.error("Gabor complexity calculation failed:", error);
      return 0;
    }
  }

  calculateTextureComplexity(texture) {
    try {
      if (!texture) return 0;

      const complexityFactors = {
        lbp: this.calculateLBPComplexity(texture.lbp || []),
        glcm: this.calculateGLCMComplexity(texture.glcm || []),
        gabor: this.calculateGaborComplexity(texture.gabor || []),
      };

      return (
        complexityFactors.lbp * 0.4 +
        complexityFactors.glcm * 0.3 +
        complexityFactors.gabor * 0.3
      );
    } catch (error) {
      console.error("Texture complexity calculation failed:", error);
      return 0;
    }
  }

  calculateTextureRegularity(texture) {
    try {
      if (!texture) return 0;

      const glcmRegularity = this.calculateGLCMRegularity(texture.glcm || []);
      const lbpRegularity = this.calculateLBPRegularity(texture.lbp || []);
      const gaborRegularity = this.calculateGaborRegularity(
        texture.gabor || []
      );

      return glcmRegularity * 0.4 + lbpRegularity * 0.3 + gaborRegularity * 0.3;
    } catch (error) {
      console.error("Texture regularity calculation failed:", error);
      return 0;
    }
  }

  calculateTextureOrientation(texture) {
    try {
      if (!texture || !texture.gabor || !texture.gabor.length) return 0;

      const orientations = texture.gabor
        .map((response) => response.orientation || 0)
        .filter((o) => !isNaN(o));

      if (orientations.length === 0) return 0;

      // Calculate circular mean of orientations
      const sumSin = orientations.reduce(
        (sum, angle) => sum + Math.sin(angle),
        0
      );
      const sumCos = orientations.reduce(
        (sum, angle) => sum + Math.cos(angle),
        0
      );

      return Math.atan2(sumSin, sumCos);
    } catch (error) {
      console.error("Texture orientation calculation failed:", error);
      return 0;
    }
  }

  // Add these helper methods for texture feature calculation
  calculateGLCMRegularity(glcmFeatures) {
    if (!Array.isArray(glcmFeatures) || glcmFeatures.length === 0) return 0;

    const homogeneities = glcmFeatures.map((f) => f.homogeneity || 0);
    const mean =
      homogeneities.reduce((a, b) => a + b, 0) / homogeneities.length;
    const variance =
      homogeneities.reduce((sum, h) => sum + Math.pow(h - mean, 2), 0) /
      homogeneities.length;

    return Math.exp(-variance * 2); // Higher regularity = lower variance
  }

  calculateLBPRegularity(lbpFeatures) {
    if (!Array.isArray(lbpFeatures) || lbpFeatures.length === 0) return 0;

    const patterns = lbpFeatures.map((f) => f.histogram || []).flat();
    if (patterns.length === 0) return 0;

    const sum = patterns.reduce((a, b) => a + b, 0);
    if (sum === 0) return 0;

    // Calculate normalized entropy as measure of regularity
    const entropy = patterns.reduce((e, count) => {
      const p = count / sum;
      return e - (p > 0 ? p * Math.log2(p) : 0);
    }, 0);

    return 1 - entropy / Math.log2(patterns.length);
  }

  calculateGaborRegularity(gaborFeatures) {
    if (!Array.isArray(gaborFeatures) || gaborFeatures.length === 0) return 0;

    const responses = gaborFeatures.map((f) => f.magnitude || 0);
    const mean = responses.reduce((a, b) => a + b, 0) / responses.length;
    const variance =
      responses.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) /
      responses.length;

    return 1 / (1 + Math.sqrt(variance) / mean);
  }

  // Enhanced cell texture extraction
  async extractCellTexture(textureRegions, x, y, width, height) {
    try {
      // Ensure textureRegions is properly formatted
      const processedRegions = Array.isArray(textureRegions)
        ? textureRegions
        : [];

      const cellFeatures = {
        glcm: [],
        lbp: [],
        gabor: [],
        statistics: {},
      };

      const cellRect = { x, y, width, height };

      // Process each valid region
      for (const region of processedRegions) {
        if (region && this.regionOverlapsCell(region, cellRect)) {
          if (region.glcm) cellFeatures.glcm.push(region.glcm);
          if (region.lbp) cellFeatures.lbp.push(region.lbp);
          if (region.gabor) cellFeatures.gabor.push(region.gabor);
        }
      }

      // Calculate statistics if we have features
      if (
        cellFeatures.glcm.length ||
        cellFeatures.lbp.length ||
        cellFeatures.gabor.length
      ) {
        cellFeatures.statistics = this.calculateTextureStatistics(cellFeatures);
      }

      return cellFeatures;
    } catch (error) {
      console.error("Cell texture extraction failed:", error);
      return {
        glcm: [],
        lbp: [],
        gabor: [],
        statistics: {},
      };
    }
  }

  async _safeColorAnalysis(buffer) {
    try {
      if (!this.colorAnalyzer) {
        this.colorAnalyzer = new ColorAnalysis(this.config.COLOR);
      }

      // Ensure we're passing a valid buffer
      if (!Buffer.isBuffer(buffer)) {
        throw new Error("Invalid input: Buffer expected");
      }

      const analysis = await this.colorAnalyzer.analyzeImage(
        Buffer.from(buffer)
      );
      return analysis || this.getDefaultColorAnalysis();
    } catch (error) {
      console.error("Color analysis failed:", error);
      return this.getDefaultColorAnalysis();
    }
  }

  getDefaultColorAnalysis() {
    return {
      statistics: {
        mean: new Float32Array(3),
        variance: new Float32Array(3),
        skewness: new Float32Array(3),
        kurtosis: new Float32Array(3),
      },
      distribution: {
        histograms: [
          new Float32Array(256),
          new Float32Array(256),
          new Float32Array(256),
        ],
        entropy: 0,
        uniformity: 0,
      },
      quality: {
        overall: 0,
        components: {
          statistics: 0,
          distribution: 0,
          harmony: 0,
        },
      },
    };
  }

  // Update spatial verification methods
  async generateSpatialVerification(buffer) {
    try {
      const { data, info } = await sharp(buffer, { failOnError: false })
        .raw()
        .toBuffer({ resolveWithObject: true });

      const colorLayout = await this.extractColorLayout(data, info);
      const edgeMap = await this.extractEdgeMap(data, info);
      const textureRegions = await this.extractTextureRegions(data, info);

      const spatialFeatures = {
        grid: await this.generateSpatialGrid(
          data,
          info,
          colorLayout,
          edgeMap,
          textureRegions
        ),
        regions: await this.analyzeSpatialRegions(
          colorLayout,
          edgeMap,
          textureRegions
        ),
        relationships: this.analyzeSpatialRelationships(
          colorLayout,
          edgeMap,
          textureRegions
        ),
        invariants: this.calculateSpatialInvariants(
          colorLayout,
          edgeMap,
          textureRegions
        ),
      };

      return {
        ...spatialFeatures,
        metadata: this.generateSpatialMetadata(spatialFeatures),
        quality: this.assessSpatialQuality(spatialFeatures),
      };
    } catch (error) {
      console.error("Spatial verification generation failed:", error);
      return this.getDefaultSpatialFeatures();
    }
  }

  getDefaultVerificationFeatures() {
    return {
      colorVerification: {},
      edgeVerification: {},
      textureVerification: {},
      spatialRelations: {},
      consistency: {},
      quality: {
        reliability: 0,
        confidence: 0,
      },
    };
  }

  computeGLCM(data, width, height, distance, angle) {
    const glcm = Array(256)
      .fill()
      .map(() => Array(256).fill(0));
    const angleRad = (angle * Math.PI) / 180;
    const dx = Math.round(Math.cos(angleRad) * distance);
    const dy = Math.round(Math.sin(angleRad) * distance);

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const nx = x + dx;
        const ny = y + dy;

        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
          const i = data[y * width + x];
          const j = data[ny * width + nx];
          glcm[i][j]++;
        }
      }
    }

    return glcm;
  }

  normalizeGLCM(glcm) {
    const normalized = Array(256)
      .fill()
      .map(() => Array(256).fill(0));
    let sum = 0;

    // Calculate sum
    for (let i = 0; i < 256; i++) {
      for (let j = 0; j < 256; j++) {
        sum += glcm[i][j];
      }
    }

    // Normalize
    if (sum > 0) {
      for (let i = 0; i < 256; i++) {
        for (let j = 0; j < 256; j++) {
          normalized[i][j] = glcm[i][j] / sum;
        }
      }
    }

    return normalized;
  }

  calculateGLCMFeatures(normalizedGLCM) {
    const features = {
      contrast: 0,
      dissimilarity: 0,
      homogeneity: 0,
      energy: 0,
      correlation: 0,
      entropy: 0,
    };

    let meanI = 0,
      meanJ = 0;
    let stdI = 0,
      stdJ = 0;

    // Calculate means
    for (let i = 0; i < 256; i++) {
      for (let j = 0; j < 256; j++) {
        meanI += i * normalizedGLCM[i][j];
        meanJ += j * normalizedGLCM[i][j];
      }
    }

    // Calculate standard deviations
    for (let i = 0; i < 256; i++) {
      for (let j = 0; j < 256; j++) {
        stdI += Math.pow(i - meanI, 2) * normalizedGLCM[i][j];
        stdJ += Math.pow(j - meanJ, 2) * normalizedGLCM[i][j];
      }
    }

    stdI = Math.sqrt(stdI);
    stdJ = Math.sqrt(stdJ);

    // Calculate features
    for (let i = 0; i < 256; i++) {
      for (let j = 0; j < 256; j++) {
        const p = normalizedGLCM[i][j];
        if (p > 0) {
          features.contrast += Math.pow(i - j, 2) * p;
          features.dissimilarity += Math.abs(i - j) * p;
          features.homogeneity += p / (1 + Math.pow(i - j, 2));
          features.energy += p * p;
          features.entropy -= p * Math.log(p);

          if (stdI > 0 && stdJ > 0) {
            features.correlation +=
              ((i - meanI) * (j - meanJ) * p) / (stdI * stdJ);
          }
        }
      }
    }

    return features;
  }

  // Add these utility and feature extraction functions to EnhancedSignatureGenerator class

  // ====== Statistical Utility Functions ======

  calculateWeightedMean(values, weights) {
    if (!values || !weights || values.length === 0) return 0;
    const weightSum = weights.reduce((sum, w) => sum + w, 0);
    const weightedSum = values.reduce(
      (sum, value, i) => sum + value * weights[i],
      0
    );
    return weightSum > 0 ? weightedSum / weightSum : 0;
  }

  calculateStandardDeviation(values, mean) {
    if (!values || values.length === 0) return 0;
    mean = mean ?? this.calculateMean(values);
    const variance =
      values.reduce((sum, value) => sum + Math.pow(value - mean, 2), 0) /
      values.length;
    return Math.sqrt(variance);
  }

  // ====== LBP Functions ======
  analyzeLBPPatterns(lbpImage) {
    try {
      const patterns = new Map();
      let totalPatterns = 0;

      // Count pattern occurrences
      for (let i = 0; i < lbpImage.length; i++) {
        const pattern = lbpImage[i];
        patterns.set(pattern, (patterns.get(pattern) || 0) + 1);
        totalPatterns++;
      }

      // Find uniform patterns
      const uniformPatterns = new Set();
      for (const [pattern] of patterns) {
        if (this.isUniformPattern(pattern)) {
          uniformPatterns.add(pattern);
        }
      }

      return {
        distribution: this.calculatePatternDistribution(
          patterns,
          totalPatterns
        ),
        uniformity: uniformPatterns.size / patterns.size,
        dominantPatterns: this.findDominantPatterns(patterns, 5),
        patterns: patterns, // Return the Map directly
      };
    } catch (error) {
      console.error("Error in analyzeLBPPatterns:", error);
      return {
        distribution: new Map(),
        uniformity: 0,
        dominantPatterns: [],
        patterns: new Map(),
      };
    }
  }

  isUniformPattern(pattern) {
    let transitions = 0;
    let lastBit = pattern & 1;

    // Count transitions in the binary pattern
    for (let i = 1; i < 8; i++) {
      const bit = (pattern >> i) & 1;
      if (bit !== lastBit) transitions++;
      lastBit = bit;
    }
    // Check last-to-first transition
    if (lastBit !== (pattern & 1)) transitions++;

    return transitions <= 2;
  }

  calculatePatternDistribution(patterns, total) {
    const distribution = new Map();
    for (const [pattern, count] of patterns.entries()) {
      distribution.set(pattern, count / total);
    }
    return distribution;
  }

  findDominantPatterns(patterns, count) {
    return Array.from(patterns.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, count)
      .map(([pattern, frequency]) => ({ pattern, frequency }));
  }

  calculatePatternStatistics(patterns) {
    const values = Array.from(patterns.values());
    return {
      mean: this.calculateMean(values),
      std: this.calculateStandardDeviation(values),
      entropy: this.calculatePatternEntropy(patterns),
      variance: this.calculatePatternVariance(values),
    };
  }

  // ====== Hash Analysis Functions ======
  analyzePatternDiversity(hash) {
    const patterns = new Set();
    const windowSize = 2; // Size of pattern window

    // Analyze bit patterns
    for (let i = 0; i < hash.length - windowSize + 1; i++) {
      let pattern = 0;
      for (let j = 0; j < windowSize; j++) {
        pattern = (pattern << 8) | hash[i + j];
      }
      patterns.add(pattern);
    }

    // Calculate diversity metrics
    const maxPatterns =
      (hash.length - windowSize + 1) * Math.pow(2, windowSize * 8);
    const diversity = patterns.size / maxPatterns;

    return {
      diversity,
      uniquePatterns: patterns.size,
      coverage: this.calculatePatternCoverage(patterns, windowSize * 8),
      distribution: this.analyzePatternDistribution(Array.from(patterns)),
    };
  }

  calculatePatternCoverage(patterns, bitLength) {
    const maxValue = Math.pow(2, bitLength);
    return patterns.size / maxValue;
  }

  analyzePatternDistribution(patterns) {
    if (patterns.length === 0) return { entropy: 0, uniformity: 1 };

    const histogram = new Array(256).fill(0);
    patterns.forEach((pattern) => {
      histogram[pattern & 0xff]++;
    });

    const probability = histogram.map((count) => count / patterns.length);
    const entropy = -probability.reduce(
      (sum, p) => sum + (p > 0 ? p * Math.log2(p) : 0),
      0
    );
    const uniformity = Math.pow(
      probability.reduce((sum, p) => sum + p * p, 0),
      0.5
    );

    return { entropy, uniformity };
  }

  // ====== Quality Assessment Functions ======
  assessWaveletHashQuality(coefficients) {
    const energyDistribution = this.analyzeWaveletEnergy(coefficients);
    const stabilityScore = this.assessWaveletStability(coefficients);
    const compressionRatio =
      this.calculateWaveletCompressionRatio(coefficients);

    return {
      quality:
        energyDistribution.score * 0.4 +
        stabilityScore * 0.3 +
        compressionRatio * 0.3,
      components: {
        energyDistribution,
        stability: stabilityScore,
        compression: compressionRatio,
      },
    };
  }

  assessRadonHashQuality(transform) {
    const coverage = this.analyzeRadonCoverage(transform);
    const stability = this.assessRadonStability(transform);
    const distinctiveness = this.calculateRadonDistinctiveness(transform);

    return {
      quality: coverage * 0.4 + stability * 0.3 + distinctiveness * 0.3,
      components: {
        coverage,
        stability,
        distinctiveness,
      },
    };
  }

  // ====== Feature Extraction Statistics ======
  extractWaveletStatistics(coefficients) {
    const statistics = {};
    const bands = ["LL", "LH", "HL", "HH"];

    for (const band of bands) {
      const values = coefficients[band];
      statistics[band] = {
        mean: this.calculateMean(values),
        std: this.calculateStandardDeviation(values),
        energy: this.calculateBandEnergy(values),
        entropy: this.calculateBandEntropy(values),
      };
    }

    return {
      bandStatistics: statistics,
      globalStatistics: this.calculateGlobalWaveletStatistics(coefficients),
      interscaleStatistics: this.calculateInterscaleStatistics(coefficients),
    };
  }

  // ====== Default Feature Sets ======

  // ====== Color Analysis Functions ======
  analyzeColorHarmony(data, info) {
    const harmonies = {
      complementary: this.findComplementaryColors(data, info),
      analogous: this.findAnalogousColors(data, info),
      triadic: this.findTriadicColors(data, info),
      splitComplementary: this.findSplitComplementaryColors(data, info),
      tetradic: this.findTetradicColors(data, info),
    };

    return {
      harmonies,
      quality: this.assessColorHarmonyQuality(harmonies),
      balance: this.analyzeColorBalance(data, info),
      contrast: this.analyzeColorContrast(data, info),
    };
  }

  // ====== Edge Detection Functions ======

  // Add these additional feature extraction functions to EnhancedSignatureGenerator class

  // ====== Statistical Functions ======

  // ====== Pattern Analysis Functions ======
  calculatePatternEntropy(patterns) {
    let totalCount = 0;
    for (const count of patterns.values()) {
      totalCount += count;
    }

    let entropy = 0;
    for (const count of patterns.values()) {
      const probability = count / totalCount;
      if (probability > 0) {
        entropy -= probability * Math.log2(probability);
      }
    }
    return entropy;
  }

  calculatePatternVariance(values) {
    const mean = this.calculateMean(values);
    return this.calculateMean(values.map((v) => Math.pow(v - mean, 2)));
  }

  // ====== Wavelet Analysis Functions ======
  analyzeWaveletEnergy(coefficients) {
    try {
      if (!coefficients || typeof coefficients !== "object") {
        return {
          distribution: new Map(),
          score: 0,
          concentration: 0,
        };
      }

      let values = [];
      Object.values(coefficients).forEach((band) => {
        if (Array.isArray(band) || ArrayBuffer.isView(band)) {
          values = values.concat(Array.from(band));
        }
      });

      if (values.length === 0) {
        return {
          distribution: new Map(),
          score: 0,
          concentration: 0,
        };
      }

      const energy =
        values.reduce((sum, val) => sum + val * val, 0) / values.length;
      return {
        distribution: new Map([["total", energy]]),
        score: Math.min(1, energy / 255),
        concentration: this.calculateEnergyConcentration(values),
      };
    } catch (error) {
      console.error("Wavelet energy analysis failed:", error);
      return {
        distribution: new Map(),
        score: 0,
        concentration: 0,
      };
    }
  }

  // ====== Hash Quality Assessment Functions ======
  calculateHashDistinctiveness(hash) {
    const patterns = this.analyzeHashPatterns(hash);
    const distribution = this.calculatePatternDistribution(patterns);

    return {
      uniqueness: this.calculatePatternUniqueness(patterns),
      entropy: this.calculatePatternEntropy(patterns),
      distribution: distribution,
      score: this.calculateDistinctivenessScore(patterns, distribution),
    };
  }

  analyzeHashPatterns(hash) {
    const patterns = new Map();
    const windowSize = 2;

    for (let i = 0; i < hash.length - windowSize + 1; i++) {
      let pattern = 0;
      for (let j = 0; j < windowSize; j++) {
        pattern = (pattern << 8) | hash[i + j];
      }
      patterns.set(pattern, (patterns.get(pattern) || 0) + 1);
    }

    return patterns;
  }

  calculateRadonCoverage(transform) {
    if (!transform || !Array.isArray(transform)) {
      return 0;
    }

    const values = Array.from(transform);
    const threshold = 0.1 * Math.max(...values);
    const significantPoints = values.filter((v) => v > threshold).length;
    return significantPoints / values.length;
  }

  calculateDistinctivenessScore(patterns, distribution) {
    const entropy = this.calculatePatternEntropy(patterns);
    const uniformity = this.calculateDistributionUniformity(distribution);
    return entropy * 0.6 + uniformity * 0.4;
  }

  // ====== Radon Transform Analysis Functions ======
  analyzeRadonCoverage(transform) {
    const coverage = this.calculateRadonCoverage(transform);
    const uniformity = this.calculateRadonUniformity(transform);
    const consistency = this.calculateRadonConsistency(transform);

    return {
      coverageRatio: coverage,
      uniformityScore: uniformity,
      consistencyScore: consistency,
      overall: coverage * 0.4 + uniformity * 0.3 + consistency * 0.3,
    };
  }

  calculateRadonUniformity(transform) {
    const mean = this.calculateMean(transform);
    const std = this.calculateStd(transform);
    return 1 / (1 + std / mean);
  }

  calculateRadonConsistency(transform) {
    const differences = [];
    for (let i = 1; i < transform.length; i++) {
      differences.push(Math.abs(transform[i] - transform[i - 1]));
    }
    return 1 / (1 + this.calculateMean(differences));
  }

  // ====== Energy Analysis Functions ======
  calculateBandEnergy(values) {
    return (
      values.reduce((energy, value) => energy + value * value, 0) /
      values.length
    );
  }

  calculateBandEntropy(values) {
    const histogram = new Float32Array(256);
    let max = -Infinity,
      min = Infinity;

    // Find range
    for (const value of values) {
      max = Math.max(max, value);
      min = Math.min(min, value);
    }

    // Build histogram
    const range = max - min;
    for (const value of values) {
      const bin = Math.min(255, Math.floor(((value - min) / range) * 255));
      histogram[bin]++;
    }

    // Calculate entropy
    let entropy = 0;
    const totalCount = values.length;
    for (const count of histogram) {
      if (count > 0) {
        const p = count / totalCount;
        entropy -= p * Math.log2(p);
      }
    }

    return entropy;
  }

  // ====== Color Analysis Functions ======
  findComplementaryColors(data, info) {
    const dominantColors = this.extractDominantColors(data, info);
    const complementaryPairs = [];

    for (const color of dominantColors) {
      const complement = this.calculateComplementaryColor(color);
      const matchingColor = this.findClosestColor(complement, dominantColors);
      if (matchingColor) {
        complementaryPairs.push([color, matchingColor]);
      }
    }

    return complementaryPairs;
  }

  calculateComplementaryColor(color) {
    return {
      r: 255 - color.r,
      g: 255 - color.g,
      b: 255 - color.b,
    };
  }

  findClosestColor(target, colors) {
    let minDistance = Infinity;
    let closest = null;

    for (const color of colors) {
      const distance = this.calculateColorDistance(target, color);
      if (distance < minDistance) {
        minDistance = distance;
        closest = color;
      }
    }

    return closest;
  }

  // ====== Gradient Analysis Functions ======
  calculateGradients(data, width) {
    const height = Math.floor(data.length / width);
    const magnitude = new Float32Array(data.length);
    const direction = new Float32Array(data.length);

    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const idx = y * width + x;

        // Sobel operators
        const gx =
          (-1 * data[idx - width - 1] +
            1 * data[idx - width + 1] +
            -2 * data[idx - 1] +
            2 * data[idx + 1] +
            -1 * data[idx + width - 1] +
            1 * data[idx + width + 1]) /
          8;

        const gy =
          (-1 * data[idx - width - 1] +
            -2 * data[idx - width] +
            -1 * data[idx - width + 1] +
            1 * data[idx + width - 1] +
            2 * data[idx + width] +
            1 * data[idx + width + 1]) /
          8;

        magnitude[idx] = Math.sqrt(gx * gx + gy * gy);
        direction[idx] = Math.atan2(gy, gx);
      }
    }

    return [magnitude, direction];
  }

  // ====== Spatial Analysis Functions ======

  async getImageDimensions(features) {
    // Extract dimensions from color features
    const width = features.width || 256; // Default fallback
    const height = features.height || 256; // Default fallback
    return { width, height };
  }

  async analyzeGridCell(features, x, y, gridSize) {
    // Analyze color features within the grid cell
    return {
      dominantColor: await this.extractCellDominantColor(
        features,
        x,
        y,
        gridSize
      ),
      colorVariance: await this.calculateCellColorVariance(
        features,
        x,
        y,
        gridSize
      ),
      edgeStrength: await this.calculateCellEdgeStrength(
        features,
        x,
        y,
        gridSize
      ),
    };
  }

  calculateGLCMStatistics(features) {
    const statistics = {};
    const featureNames = [
      "contrast",
      "dissimilarity",
      "homogeneity",
      "energy",
      "correlation",
      "entropy",
    ];

    for (const feature of featureNames) {
      const values = [];
      for (const distance in features) {
        for (const angle in features[distance]) {
          values.push(features[distance][angle][feature]);
        }
      }

      statistics[feature] = {
        mean: this.calculateMean(values),
        std: this.calculateStd(values),
        min: Math.min(...values),
        max: Math.max(...values),
        range: Math.max(...values) - Math.min(...values),
      };
    }

    return statistics;
  }

  calculateResponseQuality(features) {
    if (!features || typeof features !== "object") {
      return 0;
    }

    const metrics = [
      features.energy || 0,
      features.mean || 0,
      features.std || 0,
      features.entropy || 0,
    ];

    const weights = [0.4, 0.3, 0.2, 0.1];
    let qualityScore = 0;

    for (let i = 0; i < metrics.length; i++) {
      // Normalize each metric to [0,1] range
      const normalizedMetric = metrics[i] / (1 + metrics[i]);
      qualityScore += normalizedMetric * weights[i];
    }

    return Math.max(0, Math.min(1, qualityScore));
  }

  assessRotationStability(hash, rotations = [1, 2, 5, 10, 15]) {
    if (!hash || !hash.length) return 0;

    const similarities = rotations.map((angle) => {
      const rotated = this.simulateRotation(hash, angle);
      const similarity = this.calculateHashSimilarity(hash, rotated);
      return similarity;
    });

    return (
      similarities.reduce((acc, val) => acc + val, 0) / similarities.length
    );
  }

  assessGeometricStability(hash) {
    if (!hash || !hash.length) {
      return {
        rotation: 0,
        scale: 0,
        translation: 0,
        overall: 0,
      };
    }

    const rotationStability = this.assessRotationStability(hash);
    const scaleStability = this.assessScaleStability(hash);
    const translationStability = this.assessTranslationStability(hash);

    const overall =
      rotationStability * 0.4 +
      scaleStability * 0.3 +
      translationStability * 0.3;

    return {
      rotation: rotationStability,
      scale: scaleStability,
      translation: translationStability,
      overall,
    };
  }

  assessGaborQuality(features) {
    if (!features || !features.features) return 0;

    const responseQualities = features.features.map((feature) =>
      this.calculateResponseQuality(feature)
    );

    if (responseQualities.length === 0) return 0;

    const meanQuality =
      responseQualities.reduce((a, b) => a + b, 0) / responseQualities.length;
    const variance =
      responseQualities.reduce(
        (acc, val) => acc + Math.pow(val - meanQuality, 2),
        0
      ) / responseQualities.length;

    // Penalize high variance in response qualities
    return meanQuality * (1 - Math.sqrt(variance));
  }

  getDefaultGaborFeatures() {
    return {
      features: [],
      responses: [],
      statistics: {
        mean: 0,
        std: 0,
        energy: 0,
      },
      quality: {
        quality: 0,
        reliability: 0,
      },
    };
  }

  createGaborKernel(theta, frequency) {
    const size = 31;
    const sigma = 4.0;
    const gamma = 0.5;
    const psi = 0;
    const kernel = new Float32Array(size * size);
    const center = Math.floor(size / 2);

    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const xDash =
          (x - center) * Math.cos(theta) + (y - center) * Math.sin(theta);
        const yDash =
          -(x - center) * Math.sin(theta) + (y - center) * Math.cos(theta);

        const gaussian = Math.exp(
          -(xDash * xDash + gamma * gamma * yDash * yDash) / (2 * sigma * sigma)
        );
        const sinusoid = Math.cos(2 * Math.PI * frequency * xDash + psi);

        kernel[y * size + x] = gaussian * sinusoid;
      }
    }

    // Normalize kernel
    const sum = kernel.reduce((a, b) => a + Math.abs(b), 0);
    return kernel.map((k) => k / sum);
  }

  // Add these missing function implementations to EnhancedSignatureGenerator class
  // Add these missing function implementations to EnhancedSignatureGenerator class

  padImage(data, width, height, targetSize) {
    const padded = new Float32Array(targetSize * targetSize);

    // Copy original data with padding
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        padded[y * targetSize + x] = data[y * width + x];
      }
      // Pad right edge by repeating last pixel
      for (let x = width; x < targetSize; x++) {
        padded[y * targetSize + x] = data[y * width + (width - 1)];
      }
    }
    // Pad bottom by repeating last row
    for (let y = height; y < targetSize; y++) {
      for (let x = 0; x < targetSize; x++) {
        padded[y * targetSize + x] = padded[(height - 1) * targetSize + x];
      }
    }

    return padded;
  }
  async computeWaveletTransform(data, width, height) {
    const coefficients = {
      LL: new Float32Array((width * height) / 4),
      LH: new Float32Array((width * height) / 4),
      HL: new Float32Array((width * height) / 4),
      HH: new Float32Array((width * height) / 4),
    };

    // Apply 2D Haar wavelet transform
    for (let y = 0; y < height; y += 2) {
      for (let x = 0; x < width; x += 2) {
        const i = y * width + x;
        const outputIdx = (y / 2) * (width / 2) + x / 2;

        // Calculate averages and differences
        const a = data[i];
        const b = data[i + 1];
        const c = data[i + width];
        const d = data[i + width + 1];

        coefficients.LL[outputIdx] = (a + b + c + d) / 4;
        coefficients.LH[outputIdx] = (a + b - c - d) / 4;
        coefficients.HL[outputIdx] = (a - b + c - d) / 4;
        coefficients.HH[outputIdx] = (a - b - c + d) / 4;
      }
    }

    return coefficients;
  }

  async computeRadonTransform(data, size) {
    const numAngles = 180;
    const radonTransform = new Float32Array(numAngles * size);
    const centerX = size / 2;
    const centerY = size / 2;

    for (let angle = 0; angle < numAngles; angle++) {
      const theta = (angle * Math.PI) / 180;
      const sinTheta = Math.sin(theta);
      const cosTheta = Math.cos(theta);

      for (let r = 0; r < size; r++) {
        let sum = 0;
        let count = 0;

        // Compute line integral
        for (let x = 0; x < size; x++) {
          // Calculate corresponding y coordinate
          const y = Math.round(
            (r - centerX * cosTheta) / sinTheta +
              ((x - centerX) * cosTheta) / sinTheta +
              centerY
          );

          if (y >= 0 && y < size) {
            sum += data[y * size + x];
            count++;
          }
        }

        radonTransform[angle * size + r] = count > 0 ? sum / count : 0;
      }
    }

    return radonTransform;
  }

  async generateHashFromRadon(radonTransform) {
    const hashSize = Math.ceil(this.config.HASH.HASH_SIZE / 8);
    const hash = new Uint8Array(hashSize);

    // Compute mean of Radon transform
    const mean =
      radonTransform.reduce((a, b) => a + b, 0) / radonTransform.length;

    let hashIndex = 0;
    let bitPosition = 0;

    // Generate hash bits by comparing to mean
    for (let i = 0; i < this.config.HASH.HASH_SIZE; i++) {
      const value = radonTransform[i % radonTransform.length];
      const bit = value > mean ? 1 : 0;

      hash[hashIndex] |= bit << (7 - bitPosition);

      bitPosition++;
      if (bitPosition === 8) {
        bitPosition = 0;
        hashIndex++;
      }
    }

    return hash;
  }

  async calculateColorStatistics(data, info) {
    try {
      const channels = info.channels || 4;
      const pixels = data.length / channels;
      const stats = {
        mean: new Float32Array(channels).fill(0),
        variance: new Float32Array(channels).fill(0),
        skewness: new Float32Array(channels).fill(0),
        kurtosis: new Float32Array(channels).fill(0),
      };

      // Calculate mean
      for (let c = 0; c < channels; c++) {
        let sum = 0;
        for (let i = c; i < data.length; i += channels) {
          sum += data[i];
        }
        stats.mean[c] = sum / pixels;
      }

      // Calculate higher moments
      for (let c = 0; c < channels; c++) {
        let sumVar = 0,
          sumSkew = 0,
          sumKurt = 0;
        for (let i = c; i < data.length; i += channels) {
          const delta = data[i] - stats.mean[c];
          const delta2 = delta * delta;
          sumVar += delta2;
          sumSkew += delta2 * delta;
          sumKurt += delta2 * delta2;
        }
        stats.variance[c] = sumVar / pixels;
        stats.skewness[c] =
          sumSkew / (pixels * Math.pow(stats.variance[c], 1.5));
        stats.kurtosis[c] =
          sumKurt / (pixels * stats.variance[c] * stats.variance[c]) - 3;
      }

      return stats;
    } catch (error) {
      console.error("Color statistics calculation failed:", error);
      return {
        mean: new Float32Array(4).fill(0),
        variance: new Float32Array(4).fill(0),
        skewness: new Float32Array(4).fill(0),
        kurtosis: new Float32Array(4).fill(0),
      };
    }
  }
  // Add these feature analysis implementations to EnhancedSignatureGenerator class

  // ====== Histogram Analysis Functions ======
  calculateHistogram(values, bins) {
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min || 1;
    const histogram = new Array(bins).fill(0);

    values.forEach((value) => {
      const bin = Math.min(
        bins - 1,
        Math.floor(((value - min) / range) * bins)
      );
      histogram[bin]++;
    });

    // Normalize
    const total = values.length;
    return histogram.map((count) => count / total);
  }

  // ====== LBP Analysis Functions ======
  calculateLBPUniformity(histogram) {
    let uniformPatterns = 0;
    let totalPatterns = 0;

    for (let i = 0; i < histogram.length; i++) {
      if (this.isUniformLBPPattern(i)) {
        uniformPatterns += histogram[i];
      }
      totalPatterns += histogram[i];
    }

    return totalPatterns > 0 ? uniformPatterns / totalPatterns : 0;
  }

  isUniformLBPPattern(pattern) {
    let transitions = 0;
    const binary = pattern.toString(2).padStart(8, "0");
    const circular = binary + binary[0];

    for (let i = 0; i < 8; i++) {
      if (circular[i] !== circular[i + 1]) {
        transitions++;
      }
    }

    return transitions <= 2;
  }

  // ====== Pattern Analysis Functions ======
  calculatePatternUniqueness(patterns) {
    const uniquePatterns = new Set(patterns.keys());
    const totalPossible = Math.pow(2, 16); // For 2-byte patterns
    return uniquePatterns.size / totalPossible;
  }

  // ====== Radon Analysis Functions ======
  assessRadonStability(transform) {
    const stability = {
      angular: this.calculateAngularStability(transform),
      radial: this.calculateRadialStability(transform),
      noise: this.assessNoiseStability(transform),
    };

    return (
      stability.angular * 0.4 + stability.radial * 0.3 + stability.noise * 0.3
    );
  }

  calculateAngularStability(transform) {
    const angles = this.getRadonAngles(transform);
    const variations = angles.map((angle, i) => {
      const nextAngle = angles[(i + 1) % angles.length];
      return Math.abs(angle - nextAngle);
    });

    const meanVariation = this.calculateMean(variations);
    return 1 / (1 + meanVariation);
  }

  calculateRadialStability(transform) {
    const radialProfiles = this.getRadialProfiles(transform);
    const variations = [];

    for (let i = 0; i < radialProfiles.length - 1; i++) {
      const correlation = this.calculateProfileCorrelation(
        radialProfiles[i],
        radialProfiles[i + 1]
      );
      variations.push(correlation);
    }

    return this.calculateMean(variations);
  }

  assessNoiseStability(transform) {
    const noiseLevel = 0.1;
    const trials = 5;
    let stability = 0;

    for (let i = 0; i < trials; i++) {
      const noisyTransform = this.addRadonNoise(transform, noiseLevel);
      const similarity = this.calculateRadonSimilarity(
        transform,
        noisyTransform
      );
      stability += similarity;
    }

    return stability / trials;
  }

  getRadonAngles(transform) {
    const numAngles = Math.sqrt(transform.length);
    return Array.from({ length: numAngles }, (_, i) => (i * 180) / numAngles);
  }

  getRadialProfiles(transform) {
    const size = Math.sqrt(transform.length);
    const profiles = [];

    for (let i = 0; i < size; i++) {
      const profile = transform.slice(i * size, (i + 1) * size);
      profiles.push(Array.from(profile));
    }

    return profiles;
  }

  calculateProfileCorrelation(profile1, profile2) {
    const mean1 = this.calculateMean(profile1);
    const mean2 = this.calculateMean(profile2);

    let numerator = 0;
    let denominator1 = 0;
    let denominator2 = 0;

    for (let i = 0; i < profile1.length; i++) {
      const diff1 = profile1[i] - mean1;
      const diff2 = profile2[i] - mean2;
      numerator += diff1 * diff2;
      denominator1 += diff1 * diff1;
      denominator2 += diff2 * diff2;
    }

    return numerator / Math.sqrt(denominator1 * denominator2);
  }

  addRadonNoise(transform, level) {
    const noisy = new Float32Array(transform.length);
    for (let i = 0; i < transform.length; i++) {
      const noise = (Math.random() * 2 - 1) * level * transform[i];
      noisy[i] = transform[i] + noise;
    }
    return noisy;
  }

  calculateRadonSimilarity(transform1, transform2) {
    let similarity = 0;
    const length = Math.min(transform1.length, transform2.length);

    for (let i = 0; i < length; i++) {
      similarity +=
        1 -
        Math.abs(transform1[i] - transform2[i]) /
          Math.max(Math.abs(transform1[i]), Math.abs(transform2[i]));
    }

    return similarity / length;
  }

  // ====== Wavelet Analysis Functions ======
  calculateGlobalWaveletStatistics(coefficients) {
    const allCoefficients = this.flattenWaveletCoefficients(coefficients);

    return {
      mean: this.calculateMean(allCoefficients),
      std: this.calculateStd(allCoefficients),
      energy: this.calculateBandEnergy(allCoefficients),
      entropy: this.calculateBandEntropy(allCoefficients),
      kurtosis: this.calculateKurtosis(allCoefficients),
      skewness: this.calculateSkewness(allCoefficients),
    };
  }

  flattenWaveletCoefficients(coefficients) {
    const flattened = [];
    for (const subband of Object.values(coefficients)) {
      flattened.push(...Array.from(subband));
    }
    return flattened;
  }

  // ====== Color Clustering Functions ======
  async performMeanShiftClustering(colors, numClusters) {
    const bandwidth = this.estimateBandwidth(colors);
    let centroids = this.initializeCentroids(colors, numClusters);
    let changed = true;
    const maxIterations = 100;
    let iteration = 0;

    while (changed && iteration < maxIterations) {
      changed = false;
      const newCentroids = centroids.map(() => ({ sum: [0, 0, 0], count: 0 }));

      // Assign points to nearest centroid
      for (const color of colors) {
        const centroidIndex = this.findNearestCentroid(color, centroids);
        newCentroids[centroidIndex].sum[0] += color[0];
        newCentroids[centroidIndex].sum[1] += color[1];
        newCentroids[centroidIndex].sum[2] += color[2];
        newCentroids[centroidIndex].count++;
      }

      // Update centroids
      for (let i = 0; i < centroids.length; i++) {
        if (newCentroids[i].count > 0) {
          const newCentroid = [
            newCentroids[i].sum[0] / newCentroids[i].count,
            newCentroids[i].sum[1] / newCentroids[i].count,
            newCentroids[i].sum[2] / newCentroids[i].count,
          ];

          if (!this.arraysEqual(centroids[i], newCentroid)) {
            centroids[i] = newCentroid;
            changed = true;
          }
        }
      }

      iteration++;
    }

    // Process clusters
    return this.processClusters(colors, centroids);
  }

  estimateBandwidth(colors) {
    // Use the average distance between points as bandwidth
    const sampleSize = Math.min(colors.length, 100);
    const distances = [];

    for (let i = 0; i < sampleSize; i++) {
      const point1 = colors[Math.floor(Math.random() * colors.length)];
      const point2 = colors[Math.floor(Math.random() * colors.length)];
      distances.push(this.calculateColorDistance(point1, point2));
    }

    return this.calculateMean(distances);
  }

  initializeCentroids(colors, numClusters) {
    // K-means++ initialization
    const centroids = [colors[Math.floor(Math.random() * colors.length)]];

    while (centroids.length < numClusters) {
      const distances = colors.map((color) =>
        Math.min(
          ...centroids.map((centroid) =>
            this.calculateColorDistance(color, centroid)
          )
        )
      );

      const sum = distances.reduce((a, b) => a + b, 0);
      let rand = Math.random() * sum;

      for (let i = 0; i < distances.length; i++) {
        rand -= distances[i];
        if (rand <= 0) {
          centroids.push(colors[i]);
          break;
        }
      }
    }

    return centroids;
  }

  findNearestCentroid(color, centroids) {
    let minDistance = Infinity;
    let nearestIndex = 0;

    centroids.forEach((centroid, index) => {
      const distance = this.calculateColorDistance(color, centroid);
      if (distance < minDistance) {
        minDistance = distance;
        nearestIndex = index;
      }
    });

    return nearestIndex;
  }

  processClusters(colors, centroids) {
    const clusters = centroids.map((centroid) => ({
      center: centroid,
      points: [],
      population: 0,
    }));

    // Assign points to clusters
    for (const color of colors) {
      const clusterIndex = this.findNearestCentroid(color, centroids);
      clusters[clusterIndex].points.push(color);
    }

    // Calculate populations
    const totalPoints = colors.length;
    clusters.forEach((cluster) => {
      cluster.population = cluster.points.length / totalPoints;
    });

    return clusters;
  }

  arraysEqual(arr1, arr2) {
    if (arr1.length !== arr2.length) return false;
    for (let i = 0; i < arr1.length; i++) {
      if (Math.abs(arr1[i] - arr2[i]) > 1e-10) return false;
    }
    return true;
  }

  calculateColorDistance(color1, color2) {
    return Math.sqrt(
      Math.pow(color1[0] - color2[0], 2) +
        Math.pow(color1[1] - color2[1], 2) +
        Math.pow(color1[2] - color2[2], 2)
    );
  }

  calculateHistogramEntropy(histogram) {
    try {
      let entropy = 0;
      for (let i = 0; i < histogram.length; i++) {
        if (histogram[i] > 0) {
          entropy -= histogram[i] * Math.log2(histogram[i]);
        }
      }
      return entropy;
    } catch (error) {
      console.error("Error in calculateHistogramEntropy:", error);
      return 0;
    }
  }

  async calculateHistogramUniformity(histograms) {
    const uniformity = new Float32Array(histograms.length);

    for (let c = 0; c < histograms.length; c++) {
      for (let i = 0; i < histograms[c].length; i++) {
        uniformity[c] += histograms[c][i] * histograms[c][i];
      }
    }

    return uniformity;
  }

  async assessColorAnalysisQuality(analysis) {
    const statisticsQuality = await this.assessStatisticsQuality(
      analysis.statistics
    );
    const distributionQuality = await this.assessDistributionQuality(
      analysis.distribution
    );
    const harmonyQuality = await this.assessHarmonyQuality(analysis.harmony);

    return {
      overall:
        statisticsQuality * 0.3 +
        distributionQuality * 0.4 +
        harmonyQuality * 0.3,
      components: {
        statistics: statisticsQuality,
        distribution: distributionQuality,
        harmony: harmonyQuality,
      },
    };
  }

  assessStatisticsQuality(statistics) {
    // Check for reasonable ranges and statistical validity
    const validRange =
      statistics.mean.every((m) => m >= 0 && m <= 255) &&
      statistics.variance.every((v) => v >= 0);

    const significantVariance = statistics.variance.some((v) => v > 1.0);
    const normalDistribution = this.checkNormalDistribution(statistics);

    return (
      validRange * 0.4 + significantVariance * 0.3 + normalDistribution * 0.3
    );
  }

  checkNormalDistribution(statistics) {
    // Check if the distribution is approximately normal using skewness and kurtosis
    const normalSkewness = statistics.skewness.every((s) => Math.abs(s) < 2);
    const normalKurtosis = statistics.kurtosis.every((k) => Math.abs(k) < 7);

    return normalSkewness && normalKurtosis ? 1.0 : 0.5;
  }

  applyGaborFilter(data, width, height, kernel) {
    const response = new Float32Array(width * height);
    const kernelSize = Math.sqrt(kernel.length);
    const kernelRadius = Math.floor(kernelSize / 2);

    for (let y = kernelRadius; y < height - kernelRadius; y++) {
      for (let x = kernelRadius; x < width - kernelRadius; x++) {
        let sum = 0;

        for (let ky = -kernelRadius; ky <= kernelRadius; ky++) {
          for (let kx = -kernelRadius; kx <= kernelRadius; kx++) {
            const dataIdx = (y + ky) * width + (x + kx);
            const kernelIdx =
              (ky + kernelRadius) * kernelSize + (kx + kernelRadius);
            sum += data[dataIdx] * kernel[kernelIdx];
          }
        }

        response[y * width + x] = sum;
      }
    }

    return response;
  }

  extractGaborResponse(response) {
    try {
      const features = {};
      const values = Array.from(response); // Ensure we have an array

      // Calculate statistical features without recursion
      features.mean = this.calculateMean(values);
      features.std = this.calculateStd(values);
      features.energy =
        values.reduce((sum, val) => sum + val * val, 0) / values.length;

      // Calculate histogram features safely
      const histogram = this.calculateHistogramSafe(values, 32);
      features.entropy = this.calculateHistogramEntropy(histogram);

      return features;
    } catch (error) {
      console.error("Error in extractGaborResponse:", error);
      return {
        mean: 0,
        std: 0,
        energy: 0,
        entropy: 0,
      };
    }
  }

  calculateHistogramSafe(values, bins = 32) {
    try {
      const histogram = new Float32Array(bins).fill(0);
      if (!values || values.length === 0) return histogram;

      // Find min and max without spread operator to avoid stack issues
      let min = values[0],
        max = values[0];
      for (let i = 1; i < values.length; i++) {
        if (values[i] < min) min = values[i];
        if (values[i] > max) max = values[i];
      }

      const range = max - min || 1; // Prevent division by zero

      // Calculate histogram
      for (let i = 0; i < values.length; i++) {
        const bin = Math.min(
          bins - 1,
          Math.floor(((values[i] - min) / range) * (bins - 1))
        );
        histogram[bin]++;
      }

      // Normalize
      const total = values.length;
      for (let i = 0; i < bins; i++) {
        histogram[i] /= total;
      }

      return histogram;
    } catch (error) {
      console.error("Error in calculateHistogramSafe:", error);
      return new Float32Array(bins).fill(1 / bins);
    }
  }

  calculateGaborStatistics(features) {
    const statistics = {
      mean: {},
      energy: {},
      entropy: {},
    };

    for (const orientation of this.config.TEXTURE.GABOR_ORIENTATIONS) {
      statistics.mean[orientation] = [];
      statistics.energy[orientation] = [];
      statistics.entropy[orientation] = [];

      for (const feature of features) {
        if (feature.orientation === orientation) {
          statistics.mean[orientation].push(feature.features.mean);
          statistics.energy[orientation].push(feature.features.energy);
          statistics.entropy[orientation].push(feature.features.entropy);
        }
      }
    }

    return {
      orientationStatistics: statistics,
      globalStatistics: this.calculateGlobalGaborStatistics(features),
    };
  }

  applyDCT(data, size) {
    const dct = new Float32Array(size * size);

    for (let v = 0; v < size; v++) {
      for (let u = 0; u < size; u++) {
        let sum = 0;

        for (let y = 0; y < size; y++) {
          for (let x = 0; x < size; x++) {
            const pixel = data[y * size + x];
            const cu = Math.cos(((2 * x + 1) * u * Math.PI) / (2 * size));
            const cv = Math.cos(((2 * y + 1) * v * Math.PI) / (2 * size));
            sum += pixel * cu * cv;
          }
        }

        const cu = u === 0 ? 1 / Math.sqrt(2) : 1;
        const cv = v === 0 ? 1 / Math.sqrt(2) : 1;
        dct[v * size + u] = (2 * cu * cv * sum) / size;
      }
    }

    return dct;
  }

  calculateDCTMedian(dct) {
    // Calculate median of low-frequency components
    const lowFreqValues = [];
    for (let y = 0; y < 8; y++) {
      for (let x = 0; x < 8; x++) {
        if (y === 0 && x === 0) continue;
        lowFreqValues.push(dct[y * this.config.HASH.DCT_SIZE + x]);
      }
    }

    lowFreqValues.sort((a, b) => a - b);
    const mid = Math.floor(lowFreqValues.length / 2);
    return lowFreqValues.length % 2 === 0
      ? (lowFreqValues[mid - 1] + lowFreqValues[mid]) / 2
      : lowFreqValues[mid];
  }

  applyErrorCorrection(hashes) {
    const correctedHashes = {};
    const blockSize = 8; // Size of error correction blocks

    for (const [type, hash] of Object.entries(hashes)) {
      correctedHashes[type] = {
        hash: this.applyReedSolomonCorrection(hash.hash, blockSize),
        quality: hash.quality,
      };
    }

    return correctedHashes;
  }

  applyReedSolomonCorrection(hash, blockSize) {
    const corrected = new Uint8Array(hash.length);

    // Process hash in blocks
    for (let i = 0; i < hash.length; i += blockSize) {
      const block = hash.slice(i, i + blockSize);
      const correctedBlock = this.reedSolomonCorrectBlock(block);
      corrected.set(correctedBlock, i);
    }

    return corrected;
  }

  reedSolomonCorrectBlock(block) {
    // Implement Reed-Solomon error correction
    // This is a simplified version - in production, use a robust RS library
    const syndromes = this.calculateSyndromes(block);
    if (this.hasErrors(syndromes)) {
      return this.correctErrors(block, syndromes);
    }
    return block;
  }

  calculateSyndromes(block) {
    const syndromes = new Array(4).fill(0); // Support up to 2 error corrections
    for (let i = 0; i < syndromes.length; i++) {
      let syndrome = 0;
      for (let j = 0; j < block.length; j++) {
        syndrome ^= this.galoisMultiply(block[j], this.galoisExp(j * (i + 1)));
      }
      syndromes[i] = syndrome;
    }
    return syndromes;
  }

  hasErrors(syndromes) {
    return syndromes.some((syndrome) => syndrome !== 0);
  }

  correctErrors(block, syndromes) {
    // Berlekamp-Massey algorithm for error location
    const errorLocator = this.findErrorLocator(syndromes);
    const errorPositions = this.findErrorPositions(errorLocator, block.length);

    // Forney algorithm for error values
    const errorValues = this.calculateErrorValues(
      syndromes,
      errorLocator,
      errorPositions
    );

    // Apply corrections
    const corrected = new Uint8Array(block);
    errorPositions.forEach((pos, idx) => {
      corrected[pos] ^= errorValues[idx];
    });

    return corrected;
  }

  findErrorLocator(syndromes) {
    // Berlekamp-Massey algorithm implementation
    const L = Math.floor(syndromes.length / 2);
    const polynomial = new Uint8Array(L + 1);
    polynomial[0] = 1;
    let currentL = 0;
    let currentDisc = 1;

    for (let n = 0; n < syndromes.length; n++) {
      let discrepancy = syndromes[n];
      for (let i = 1; i <= currentL; i++) {
        discrepancy ^= this.galoisMultiply(polynomial[i], syndromes[n - i]);
      }

      if (discrepancy !== 0) {
        const oldPoly = polynomial.slice();
        for (let i = n - currentDisc + 1; i <= n; i++) {
          if (i < 0) continue;
          const pos = i + L - n;
          if (pos >= 0) {
            polynomial[pos] ^= this.galoisMultiply(
              discrepancy,
              this.galoisInverse(currentDisc)
            );
          }
        }

        if (2 * currentL <= n) {
          currentL = n + 1 - currentL;
          currentDisc = discrepancy;
        }
      }
    }

    return polynomial.slice(0, currentL + 1);
  }

  findErrorPositions(errorLocator, messageLength) {
    const positions = [];
    for (let i = 0; i < messageLength; i++) {
      let sum = 0;
      for (let j = 0; j < errorLocator.length; j++) {
        sum ^= this.galoisMultiply(errorLocator[j], this.galoisExp(i * j));
      }
      if (sum === 0) {
        positions.push(messageLength - 1 - i);
      }
    }
    return positions;
  }

  calculateErrorValues(syndromes, errorLocator, errorPositions) {
    const values = new Array(errorPositions.length);
    const evaluator = this.calculateErrorEvaluator(syndromes, errorLocator);

    errorPositions.forEach((pos, idx) => {
      const xi = this.galoisExp(pos);
      const xiInverse = this.galoisInverse(xi);

      let evaluatorValue = 0;
      for (let j = 0; j < evaluator.length; j++) {
        evaluatorValue ^= this.galoisMultiply(
          evaluator[j],
          this.galoisExp(j * pos)
        );
      }

      let locatorDerivative = 0;
      for (let j = 0; j < errorLocator.length - 1; j++) {
        const degree = errorLocator.length - 1 - j;
        if (degree % 2 === 1) {
          locatorDerivative ^= this.galoisMultiply(
            errorLocator[j],
            this.galoisExp(degree * pos)
          );
        }
      }

      values[idx] = this.galoisMultiply(
        evaluatorValue,
        this.galoisInverse(locatorDerivative)
      );
    });

    return values;
  }

  calculateErrorEvaluator(syndromes, errorLocator) {
    const evaluator = new Uint8Array(syndromes.length);
    for (let i = 0; i < syndromes.length; i++) {
      let sum = 0;
      for (let j = 0; j <= i; j++) {
        if (j < errorLocator.length) {
          sum ^= this.galoisMultiply(errorLocator[j], syndromes[i - j]);
        }
      }
      evaluator[i] = sum;
    }
    return evaluator;
  }

  // Galois field arithmetic helpers
  galoisMultiply(a, b) {
    if (a === 0 || b === 0) return 0;
    return this.galoisExp((this.galoisLog(a) + this.galoisLog(b)) % 255);
  }

  galoisInverse(x) {
    if (x === 0) return 0;
    return this.galoisExp((255 - this.galoisLog(x)) % 255);
  }

  // Pre-computed Galois field tables
  galoisExp(x) {
    return this.GF_EXP[x % 255];
  }

  galoisLog(x) {
    return this.GF_LOG[x];
  }

  // Galois field tables initialization
  initGaloisTables() {
    this.GF_EXP = new Uint8Array(256);
    this.GF_LOG = new Uint8Array(256);
    let x = 1;
    for (let i = 0; i < 255; i++) {
      this.GF_EXP[i] = x;
      this.GF_LOG[x] = i;
      x = this.galoisMultiplyNoTable(x, 2);
    }
    this.GF_EXP[255] = this.GF_EXP[0];
  }

  galoisMultiplyNoTable(a, b) {
    let p = 0;
    for (let i = 0; i < 8; i++) {
      if ((b & 1) !== 0) {
        p ^= a;
      }
      const highBitSet = (a & 0x80) !== 0;
      a <<= 1;
      if (highBitSet) {
        a ^= 0x1b; // The primitive polynomial x^8 + x^4 + x^3 + x + 1
      }
      b >>= 1;
    }
    return p;
  }

  assessDCTHashQuality(dctHash) {
    return {
      robustness: this.calculateHashRobustness(dctHash),
      distinctiveness: this.calculateHashDistinctiveness(dctHash),
      stability: this.calculateHashStability(dctHash),
    };
  }

  assessCombinedHashQuality(hashes) {
    if (!hashes || !Object.keys(hashes).length) {
      return {
        robustness: 0,
        distinctiveness: 0,
        stability: 0,
      };
    }

    const qualities = Object.values(hashes).map(
      (hash) =>
        hash.quality || {
          robustness: 0,
          distinctiveness: 0,
          stability: 0,
        }
    );

    return {
      robustness: this.combineQualityMetrics(qualities, "robustness"),
      distinctiveness: this.combineQualityMetrics(qualities, "distinctiveness"),
      stability: this.combineQualityMetrics(qualities, "stability"),
    };
  }

  combineQualityMetrics(qualities, metric) {
    if (!qualities || !qualities.length) return 0;

    const weights = [0.4, 0.3, 0.3]; // Weights for different hash types
    return qualities.reduce((sum, quality, index) => {
      const value = quality && quality[metric] ? quality[metric] : 0;
      return sum + value * (weights[index] || 0);
    }, 0);
  }

  async extractSpatialRawData(buffer) {
    const { data, info } = await sharp(buffer)
      .raw()
      .toBuffer({ resolveWithObject: true });

    return {
      data: new Uint8Array(data),
      width: info.width,
      height: info.height,
      channels: info.channels,
    };
  }

  calculateFeatureImportance(features) {
    if (!features || typeof features !== "object") return 0;

    try {
      const importanceFactors = {
        density: 0.3,
        strength: 0.3,
        coherence: 0.2,
        distinctiveness: 0.2,
      };

      const validFactors = [];
      for (const [factor, weight] of Object.entries(importanceFactors)) {
        if (typeof features[factor] === "number") {
          validFactors.push(features[factor] * weight);
        }
      }

      return validFactors.length > 0
        ? validFactors.reduce((a, b) => a + b, 0)
        : 0;
    } catch (error) {
      console.error("Feature importance calculation failed:", error);
      return 0;
    }
  }

  calculateCellImportance(features) {
    if (!features) return 0;

    try {
      const weights = {
        color: 0.4,
        edges: 0.3,
        texture: 0.3,
      };

      let totalImportance = 0;
      let validFeatures = 0;

      for (const [type, feature] of Object.entries(features)) {
        if (feature && typeof feature === "object") {
          const featureImportance = this.calculateFeatureImportance(feature);
          if (!isNaN(featureImportance)) {
            totalImportance += featureImportance * (weights[type] || 0);
            validFeatures++;
          }
        }
      }

      return validFeatures > 0 ? totalImportance / validFeatures : 0;
    } catch (error) {
      console.error("Cell importance calculation failed:", error);
      return 0;
    }
  }

  calculateCellConsistency(features) {
    if (!features) return 0;

    try {
      const consistencyScores = [];

      if (features.color) {
        consistencyScores.push(features.color.coherence || 0);
      }
      if (features.edges) {
        consistencyScores.push(features.edges.continuity || 0);
      }
      if (features.texture) {
        consistencyScores.push(features.texture.regularity || 0);
      }

      return consistencyScores.length > 0
        ? consistencyScores.reduce((a, b) => a + b, 0) /
            consistencyScores.length
        : 0;
    } catch (error) {
      console.error("Cell consistency calculation failed:", error);
      return 0;
    }
  }

  calculateCellDistinctiveness(features) {
    if (!features) return 0;

    const colorDist = features.color?.distribution?.entropy || 0;
    const edgeDist = features.edges?.distribution?.entropy || 0;
    const textureDist = features.texture?.statistics?.entropy || 0;

    return colorDist * 0.4 + edgeDist * 0.3 + textureDist * 0.3;
  }

  analyzeSpatialCell(
    colorLayout,
    edgeMap,
    textureRegions,
    x,
    y,
    width,
    height
  ) {
    try {
      if (!colorLayout || !edgeMap || !textureRegions) {
        return this.getDefaultCellFeatures();
      }

      const features = {
        color: this.extractCellColorFeatures(colorLayout, x, y, width, height),
        edges: this.extractCellEdgeFeatures(edgeMap, x, y, width, height),
        texture: this.extractCellTextureFeatures(
          textureRegions,
          x,
          y,
          width,
          height
        ),
      };

      // Calculate combined importance
      const importance = this.calculateCellImportance(features);
      const consistency = this.calculateCellConsistency(features);
      const distinctiveness = this.calculateCellDistinctiveness(features);

      return {
        ...features,
        importance,
        consistency,
        distinctiveness,
      };
    } catch (error) {
      console.error("Spatial cell analysis failed:", error);
      return this.getDefaultCellFeatures();
    }
  }

  generateSpatialGrid(rawData, colorLayout, edgeMap, textureRegions) {
    const gridSize = 8;
    const cellWidth = Math.floor(rawData.width / gridSize);
    const cellHeight = Math.floor(rawData.height / gridSize);

    const grid = Array(gridSize)
      .fill()
      .map(() =>
        Array(gridSize)
          .fill()
          .map(() => ({
            color: null,
            edges: null,
            texture: null,
            features: null,
          }))
      );

    for (let y = 0; y < gridSize; y++) {
      for (let x = 0; x < gridSize; x++) {
        grid[y][x] = this.analyzeSpatialCell(
          rawData,
          colorLayout,
          edgeMap,
          textureRegions,
          x * cellWidth,
          y * cellHeight,
          cellWidth,
          cellHeight
        );
      }
    }

    return {
      grid,
      statistics: this.calculateGridStatistics(grid),
      relationships: this.analyzeGridRelationships(grid),
    };
  }

  combineCellFeatures(features) {
    const weights = {
      color: 0.4,
      edges: 0.3,
      texture: 0.3,
    };

    return {
      importance: this.calculateFeatureImportance(features, weights),
      consistency: this.calculateFeatureConsistency(features),
      distinctiveness: this.calculateFeatureDistinctiveness(features),
    };
  }

  calculateFeatureConsistency(features) {
    const consistencyValues = Object.values(features).map(
      (f) => f?.consistency || 0
    );

    return (
      consistencyValues.reduce((sum, val) => sum + val, 0) /
      consistencyValues.length
    );
  }

  calculateFeatureDistinctiveness(features) {
    const distinctivenessValues = Object.values(features).map(
      (f) => f?.distinctiveness || 0
    );

    return (
      distinctivenessValues.reduce((sum, val) => sum + val, 0) /
      distinctivenessValues.length
    );
  }

  calculateRelativeLuminance(color) {
    const rsRGB = color.r / 255;
    const gsRGB = color.g / 255;
    const bsRGB = color.b / 255;

    const r =
      rsRGB <= 0.03928 ? rsRGB / 12.92 : Math.pow((rsRGB + 0.055) / 1.055, 2.4);
    const g =
      gsRGB <= 0.03928 ? gsRGB / 12.92 : Math.pow((gsRGB + 0.055) / 1.055, 2.4);
    const b =
      bsRGB <= 0.03928 ? bsRGB / 12.92 : Math.pow((bsRGB + 0.055) / 1.055, 2.4);

    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
  }

  calculateColorContrastRatio(color1, color2) {
    const l1 = this.calculateRelativeLuminance(color1);
    const l2 = this.calculateRelativeLuminance(color2);
    const lighter = Math.max(l1, l2);
    const darker = Math.min(l1, l2);
    return (lighter + 0.05) / (darker + 0.05);
  }

  calculateLocalContrast(colors) {
    if (!colors || !Array.isArray(colors) || colors.length === 0) {
      return 0;
    }

    let maxContrast = 0;
    for (let i = 0; i < colors.length; i++) {
      for (let j = i + 1; j < colors.length; j++) {
        const contrast = this.calculateColorContrastRatio(colors[i], colors[j]);
        maxContrast = Math.max(maxContrast, contrast);
      }
    }
    return maxContrast;
  }

  calculateColorCoherence(colors) {
    if (!colors || !Array.isArray(colors) || colors.length === 0) {
      return 0;
    }

    try {
      let totalCoherence = 0;
      let validColors = 0;

      for (let i = 0; i < colors.length - 1; i++) {
        if (colors[i] && colors[i + 1]) {
          const similarity = this.calculateColorSimilarity(
            colors[i],
            colors[i + 1]
          );
          if (!isNaN(similarity)) {
            totalCoherence += similarity;
            validColors++;
          }
        }
      }

      return validColors > 0 ? totalCoherence / validColors : 0;
    } catch (error) {
      console.error("Color coherence calculation failed:", error);
      return 0;
    }
  }

  async extractCellColorFeatures(colorLayout, x, y, width, height) {
    try {
      if (!colorLayout) return this.getDefaultCellColorFeatures();

      const cellColors = this.extractCellColors(
        colorLayout,
        x,
        y,
        width,
        height
      );
      const dominantColors = await this.findDominantColors(cellColors, {
        width: Math.floor(colorLayout.width / width),
        height: Math.floor(colorLayout.height / height),
        channels: colorLayout.channels || 4,
      });

      return {
        dominant: dominantColors[0] || this.getDefaultDominantColor(),
        distribution: this.analyzeColorDistribution(cellColors),
        contrast: this.calculateLocalContrast(cellColors),
        coherence: this.calculateColorCoherence(cellColors),
      };
    } catch (error) {
      console.error("Cell color feature extraction failed:", error);
      return this.getDefaultCellColorFeatures();
    }
  }

  getDefaultDominantColor() {
    return {
      lab: { l: 50, a: 0, b: 0 },
      rgb: { r: 128, g: 128, b: 128 },
      population: 0,
      confidence: 0,
    };
  }

  calculateEdgeDensity(edges) {
    try {
      if (!edges || !ArrayBuffer.isView(edges)) {
        return 0;
      }

      // Convert to Array if it's a TypedArray
      const edgeArray = Array.from(edges);
      const nonZeroEdges = edgeArray.filter((e) => e > 0).length;
      return nonZeroEdges / edgeArray.length;
    } catch (error) {
      console.error("Edge density calculation failed:", error);
      return 0;
    }
  }

  extractCellEdgeFeatures(edgeMap, x, y, width, height) {
    const cellEdges = this.extractCellEdges(edgeMap, x, y, width, height);
    return {
      density: this.calculateEdgeDensity(cellEdges),
      orientation: this.analyzeEdgeOrientations(cellEdges),
      continuity: this.analyzeEdgeContinuity(cellEdges),
      strength: this.calculateEdgeStrength(cellEdges),
    };
  }

  calculateLocalGradient(edges, x, y, width) {
    const gx = (edges[y * width + (x + 1)] - edges[y * width + (x - 1)]) / 2;
    const gy = (edges[(y + 1) * width + x] - edges[(y - 1) * width + x]) / 2;
    return [gx, gy];
  }

  calculateOrientationCoherence(histogram) {
    if (!histogram || histogram.length === 0) return 0;

    const mean = histogram.reduce((a, b) => a + b, 0) / histogram.length;
    const variance =
      histogram.reduce((sum, h) => sum + Math.pow(h - mean, 2), 0) /
      histogram.length;

    return 1 - Math.sqrt(variance); // Higher coherence = lower variance
  }

  analyzeEdgeOrientations(edges, width) {
    if (!edges || !width || edges.length === 0) {
      return {
        histogram: new Float32Array(36).fill(0),
        dominant: [],
        coherence: 0,
      };
    }

    const histogram = new Float32Array(36).fill(0);
    const height = Math.floor(edges.length / width);

    // Calculate gradients
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const idx = y * width + x;
        if (edges[idx] > 0) {
          const [gx, gy] = this.calculateLocalGradient(edges, x, y, width);
          const angle = ((Math.atan2(gy, gx) * 180) / Math.PI + 360) % 360;
          const bin = Math.floor(angle / 10); // 36 bins for 360 degrees
          histogram[bin] += edges[idx];
        }
      }
    }

    // Normalize histogram
    const sum = histogram.reduce((a, b) => a + b, 0);
    if (sum > 0) {
      for (let i = 0; i < histogram.length; i++) {
        histogram[i] /= sum;
      }
    }

    // Find dominant orientations
    const dominant = this.findDominantOrientations(histogram);

    // Calculate orientation coherence
    const coherence = this.calculateOrientationCoherence(histogram);

    return {
      histogram,
      dominant,
      coherence,
    };
  }

  async extractCellTexture(textureRegions, x, y, width, height) {
    try {
      const cellFeatures = {
        glcm: [],
        lbp: [],
        gabor: [],
        statistics: {},
      };

      // Extract texture features for the cell region
      const cellRect = { x, y, width, height };

      for (const region of textureRegions) {
        if (this.regionOverlapsCell(region, cellRect)) {
          cellFeatures.glcm.push(region.glcm || {});
          cellFeatures.lbp.push(region.lbp || {});
          cellFeatures.gabor.push(region.gabor || {});
        }
      }

      return cellFeatures;
    } catch (error) {
      console.error("Cell texture extraction failed:", error);
      return {
        glcm: [],
        lbp: [],
        gabor: [],
        statistics: {},
      };
    }
  }

  regionOverlapsCell(region, cell) {
    return !(
      region.bounds.maxX < cell.x ||
      region.bounds.minX > cell.x + cell.width ||
      region.bounds.maxY < cell.y ||
      region.bounds.minY > cell.y + cell.height
    );
  }

  findDominantOrientations(histogram) {
    const peaks = [];
    const threshold = 0.1; // 10% of max value

    for (let i = 0; i < histogram.length; i++) {
      const prev = histogram[(i - 1 + histogram.length) % histogram.length];
      const next = histogram[(i + 1) % histogram.length];

      if (
        histogram[i] > threshold &&
        histogram[i] > prev &&
        histogram[i] > next
      ) {
        peaks.push({
          angle: i * 10,
          magnitude: histogram[i],
        });
      }
    }

    // Return top 3 peaks sorted by magnitude
    return peaks.sort((a, b) => b.magnitude - a.magnitude).slice(0, 3);
  }

  analyzeTextureOrientation(texture) {
    try {
      if (!texture || !texture.gabor || !texture.gabor.length) {
        return {
          dominant: [],
          coherence: 0,
          distribution: new Float32Array(36).fill(0),
        };
      }

      // Extract orientation responses
      const orientationResponses = texture.gabor
        .filter(
          (response) => response && typeof response.orientation === "number"
        )
        .map((response) => ({
          angle: response.orientation,
          magnitude: response.magnitude || 0,
        }));

      if (orientationResponses.length === 0) {
        return {
          dominant: [],
          coherence: 0,
          distribution: new Float32Array(36).fill(0),
        };
      }

      // Calculate orientation histogram
      const histogram = new Float32Array(36).fill(0);
      orientationResponses.forEach((response) => {
        const bin = Math.floor((response.angle % 360) / 10);
        histogram[bin] += response.magnitude;
      });

      // Normalize histogram
      const sum = histogram.reduce((a, b) => a + b, 0);
      if (sum > 0) {
        for (let i = 0; i < histogram.length; i++) {
          histogram[i] /= sum;
        }
      }

      // Find dominant orientations
      const dominantOrientations = this.findDominantOrientations(histogram);

      // Calculate orientation coherence
      const coherence = this.calculateOrientationCoherence(histogram);

      return {
        dominant: dominantOrientations,
        coherence,
        distribution: histogram,
      };
    } catch (error) {
      console.error("Texture orientation analysis failed:", error);
      return {
        dominant: [],
        coherence: 0,
        distribution: new Float32Array(36).fill(0),
      };
    }
  }

  handleTextureRegions(textureRegions) {
    if (!textureRegions) {
      return [];
    }

    // Convert textureRegions to array if it's not already
    const regions = Array.isArray(textureRegions)
      ? textureRegions
      : [textureRegions];

    // Filter out invalid regions and ensure required properties
    return regions
      .filter((region) => region && region.points && region.points.length > 0)
      .map((region) => ({
        ...region,
        bounds: region.bounds || this.calculateBounds(region.points),
        features: region.features || this.calculateRegionFeatures(region),
      }));
  }

  calculateRegionFeatures(region) {
    return {
      size: region.points.length,
      density: this.calculateRegionDensity(region),
      perimeter: this.calculatePerimeter(region.points),
      compactness: this.calculateRegionCompactness(region),
    };
  }

  isValidPosition(x, y, width, height) {
    return (
      typeof x === "number" &&
      typeof y === "number" &&
      !isNaN(x) &&
      !isNaN(y) &&
      x >= 0 &&
      x < width &&
      y >= 0 &&
      y < height
    );
  }

  calculateColorThreshold(baseColor) {
    if (!baseColor || typeof baseColor !== "object") {
      return 30; // Default threshold
    }

    // Adaptive threshold based on color intensity
    const intensity = (baseColor.r + baseColor.g + baseColor.b) / 3;
    return Math.max(10, Math.min(30, intensity * 0.15));
  }

  async growColorRegion(colorLayout, startX, startY, visited) {
    if (!colorLayout || !colorLayout.width || !colorLayout.height) {
      return {
        points: [],
        colors: [],
        bounds: {
          minX: 0,
          maxX: 0,
          minY: 0,
          maxY: 0,
        },
      };
    }

    const region = {
      points: [],
      colors: [],
      bounds: {
        minX: startX,
        maxX: startX,
        minY: startY,
        maxY: startY,
      },
    };

    try {
      const stack = [[startX, startY]];
      const baseColor = this.getColorAt(colorLayout, startX, startY);
      const threshold = this.calculateColorThreshold(baseColor);

      while (stack.length > 0) {
        const [x, y] = stack.pop();
        const key = `${x},${y}`;

        if (
          !this.isValidPosition(x, y, colorLayout.width, colorLayout.height) ||
          visited.has(key)
        ) {
          continue;
        }

        visited.add(key);
        const currentColor = this.getColorAt(colorLayout, x, y);

        if (this.isColorSimilar(baseColor, currentColor, threshold)) {
          region.points.push([x, y]);
          region.colors.push(currentColor);
          this.updateRegionBounds(region.bounds, x, y);
          this.addValidNeighbors(stack, x, y, colorLayout, visited);
        }
      }

      return region;
    } catch (error) {
      console.error("Error growing color region:", error);
      return {
        points: [],
        colors: [],
        bounds: {
          minX: 0,
          maxX: 0,
          minY: 0,
          maxY: 0,
        },
      };
    }
  }

  addValidNeighbors(stack, x, y, features, visited) {
    const directions = [
      [-1, 0],
      [1, 0],
      [0, -1],
      [0, 1],
      [-1, -1],
      [-1, 1],
      [1, -1],
      [1, 1],
    ];

    const width = features.width || features.info?.width;
    const height = features.height || features.info?.height;

    for (const [dx, dy] of directions) {
      const nx = x + dx;
      const ny = y + dy;

      if (
        this.isValidPosition(nx, ny, width, height) &&
        !visited.has(`${nx},${ny}`)
      ) {
        stack.push([nx, ny]);
      }
    }
  }

  analyzeColorRegion(region) {
    return {
      statistics: this.calculateRegionStatistics(region),
      shape: this.analyzeRegionShape(region),
      color: this.analyzeRegionColor(region),
      significance: this.calculateRegionSignificance(region),
    };
  }

  calculateRegionStatistics(region) {
    if (!region || !Array.isArray(region.points)) {
      return {
        size: 0,
        density: 0,
        compactness: 0,
        perimeter: 0,
      };
    }

    const points = region.points;
    const area = points.length;

    if (area === 0) {
      return {
        size: 0,
        density: 0,
        compactness: 0,
        perimeter: 0,
      };
    }

    const bounds = this.calculateBounds(points);
    const boundingArea =
      (bounds.maxX - bounds.minX + 1) * (bounds.maxY - bounds.minY + 1);
    const perimeter = this.calculatePerimeter(points);

    return {
      size: area,
      density: boundingArea > 0 ? area / boundingArea : 0,
      compactness:
        perimeter > 0 ? (4 * Math.PI * area) / (perimeter * perimeter) : 0,
      perimeter: perimeter,
    };
  }

  analyzeRegionShape(region) {
    return {
      orientation: this.calculateRegionOrientation(region),
      elongation: this.calculateRegionElongation(region),
      circularity: this.calculateRegionCircularity(region),
      convexity: this.calculateRegionConvexity(region),
    };
  }

  calculateQualityReliability(qualities) {
    if (!qualities) return 0;

    const reliabilityScores = Object.values(qualities)
      .filter(
        (q) => q && typeof q === "object" && typeof q.reliability === "number"
      )
      .map((q) => q.reliability);

    return reliabilityScores.length > 0
      ? reliabilityScores.reduce((sum, score) => sum + score, 0) /
          reliabilityScores.length
      : 0;
  }

  calculateQualityConfidence(qualities) {
    if (!qualities) return 0;

    const confidenceScores = Object.values(qualities)
      .filter(
        (q) => q && typeof q === "object" && typeof q.confidence === "number"
      )
      .map((q) => q.confidence);

    return confidenceScores.length > 0
      ? confidenceScores.reduce((sum, score) => sum + score, 0) /
          confidenceScores.length
      : 0;
  }

  calculateCombinedQuality(qualities) {
    if (!qualities || typeof qualities !== "object") {
      return {
        overall: 0,
        reliability: 0,
        confidence: 0,
      };
    }

    try {
      const weights = {
        color: 0.25,
        edge: 0.25,
        texture: 0.2,
        hash: 0.15,
        spatial: 0.15,
      };

      let overall = 0;
      let validComponents = 0;

      for (const [component, quality] of Object.entries(qualities)) {
        if (quality && typeof quality === "number") {
          overall += quality * (weights[component] || 0);
          validComponents++;
        }
      }

      const reliability = this.calculateQualityReliability(qualities);
      const confidence = this.calculateQualityConfidence(qualities);

      return {
        overall: validComponents > 0 ? overall : 0,
        reliability,
        confidence,
      };
    } catch (error) {
      console.error("Error calculating combined quality:", error);
      return {
        overall: 0,
        reliability: 0,
        confidence: 0,
      };
    }
  }

  calculateComponentsReliability(components) {
    if (!components || typeof components !== "object") return 0;

    const reliabilities = Object.values(components).filter(
      (comp) => typeof comp === "number" && !isNaN(comp)
    );

    return reliabilities.length > 0
      ? reliabilities.reduce((sum, rel) => sum + rel, 0) / reliabilities.length
      : 0;
  }

  assessSignatureReliability(qualities) {
    if (!qualities || typeof qualities !== "object") {
      return 0;
    }

    try {
      const reliabilityScores = [];
      const weights = {
        color: 0.25,
        edge: 0.25,
        texture: 0.2,
        hash: 0.15,
        spatial: 0.15,
      };

      for (const [component, quality] of Object.entries(qualities)) {
        if (quality && typeof quality === "object") {
          const componentReliability =
            this.calculateComponentReliability(quality);
          reliabilityScores.push({
            score: componentReliability,
            weight: weights[component] || 0,
          });
        }
      }

      if (reliabilityScores.length === 0) return 0;

      // Calculate weighted average
      const totalWeight = reliabilityScores.reduce(
        (sum, item) => sum + item.weight,
        0
      );
      const weightedSum = reliabilityScores.reduce(
        (sum, item) => sum + item.score * item.weight,
        0
      );

      return totalWeight > 0 ? weightedSum / totalWeight : 0;
    } catch (error) {
      console.error("Error assessing signature reliability:", error);
      return 0;
    }
  }

  calculateComponentConfidence(quality) {
    if (!quality || typeof quality !== "object") return 0;

    // Collect all available confidence metrics
    const metrics = [
      quality.confidence,
      quality.reliability,
      quality.quality,
      quality.robustness,
      quality.stability,
    ].filter((metric) => typeof metric === "number" && !isNaN(metric));

    // Return average of available metrics
    return metrics.length > 0
      ? metrics.reduce((sum, metric) => sum + metric, 0) / metrics.length
      : 0;
  }

  async compareSignatures(signature1, signature2) {
    const comparisons = await Promise.all([
      this.compareColorFeatures(signature1.colors, signature2.colors),
      this.compareEdgeFeatures(signature1.edges, signature2.edges),
      this.compareTextureFeatures(signature1.textures, signature2.textures),
      this.compareHashFeatures(signature1.hashes, signature2.hashes),
      this.compareSpatialFeatures(
        signature1.spatialVerification,
        signature2.spatialVerification
      ),
    ]);

    const [
      colorSimilarity,
      edgeSimilarity,
      textureSimilarity,
      hashSimilarity,
      spatialSimilarity,
    ] = comparisons;

    // Calculate weighted similarity score
    const weightedScore = this.calculateWeightedSimilarity({
      color: colorSimilarity,
      edge: edgeSimilarity,
      texture: textureSimilarity,
      hash: hashSimilarity,
      spatial: spatialSimilarity,
    });

    return {
      similarity: weightedScore,
      components: {
        color: colorSimilarity,
        edge: edgeSimilarity,
        texture: textureSimilarity,
        hash: hashSimilarity,
        spatial: spatialSimilarity,
      },
      confidence: this.calculateComparisonConfidence(comparisons),
      details: this.generateComparisonDetails(comparisons),
    };
  }

  calculateWeightedSimilarity(similarities) {
    const weights = {
      color: 0.25,
      edge: 0.25,
      texture: 0.2,
      hash: 0.15,
      spatial: 0.15,
    };

    return Object.entries(similarities).reduce(
      (score, [key, similarity]) => score + similarity * weights[key],
      0
    );
  }

  calculateComparisonConfidence(comparisons) {
    return {
      overall: this.calculateOverallConfidence(comparisons),
      components: this.calculateComponentConfidence(comparisons),
      reliability: this.assessComparisonReliability(comparisons),
    };
  }

  static async createIndex(signatures) {
    const index = new ImageIndex();
    for (const signature of signatures) {
      await index.addSignature(signature);
    }
    return index;
  }

  static async findMatches(signature, index, options = {}) {
    const matches = await index.search(signature, options);
    return this.rankMatches(matches, signature, options);
  }
}

module.exports = {
  BufferManager,
  EnhancedSignatureGenerator,
};

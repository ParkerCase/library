// textureAnalysis.js - Enhanced Production Version
const sharp = require('sharp');
const _ = require('lodash');
const mathUtils = require('./mathUtils');
const monitor = require('./monitoring');
const { ValidationError, ProcessingError } = require('./errorHandler');


class TextureAnalysis {
    // Constants for texture analysis
    static GABOR_ORIENTATIONS = [0, 30, 60, 90, 120, 150];
    static GABOR_FREQUENCIES = [0.1, 0.2, 0.3, 0.4];
    static GABOR_SIGMA = 3.0;
    static GABOR_GAMMA = 0.5;

    // Constants for multi-scale analysis
    static SCALE_FACTORS = [1, 0.5, 0.25, 0.125];
    static WINDOW_SIZES = [3, 5, 7, 9];

static async analyzeTexture(buffer) {
    const opId = monitor.startOperation('textureAnalysis');
    
    try {
        if (!Buffer.isBuffer(buffer)) {
            throw new ValidationError('Invalid input: Buffer expected');
        }

        // Create multi-scale pyramid
        const scales = await this.createMultiResolutionAnalysis(buffer);

        // Analyze each scale with error handling
        const scaleAnalyses = await Promise.allSettled(
            scales.map(async (scaleBuffer, index) => ({
                scale: Math.pow(2, index),
                ...(await this._safeAnalyze(scaleBuffer))
            }))
        );

        // Process results with fallbacks
        const validAnalyses = scaleAnalyses
            .filter(result => result.status === 'fulfilled')
            .map(result => result.value);

        return {
            version: '2.0',
            timestamp: Date.now(),
            multiResolution: validAnalyses,
            globalFeatures: this.combineScaleFeatures(validAnalyses),
            statistics: this.calculateTextureStatistics(validAnalyses),
            patterns: await this.detectTexturePatterns(buffer),
            quality: {
                completeness: validAnalyses.length / scales.length,
                reliability: this.assessAnalysisReliability(validAnalyses)
            }
        };

    } catch (error) {
        if (error instanceof ValidationError) {
            throw error;
        }
        throw new ProcessingError('Texture analysis failed', { error });
    } finally {
        monitor.endOperation(opId);
    }
}

    static async createMultiResolutionAnalysis(buffer) {
        const scales = [buffer];
        let currentBuffer = buffer;

        try {
            const metadata = await sharp(buffer).metadata();
            const minDimension = Math.min(metadata.width, metadata.height);

            // Create scale pyramid with size validation
            for (let factor of this.SCALE_FACTORS.slice(1)) {
                const targetSize = Math.round(minDimension * factor);
                if (targetSize < 16) break; // Minimum size threshold

                currentBuffer = await sharp(currentBuffer)
                    .resize({
                        width: Math.round(metadata.width * factor),
                        height: Math.round(metadata.height * factor),
                        kernel: sharp.kernel.lanczos3,
                        fastShrinkOnLoad: true
                    })
                    .toBuffer();

                scales.push(currentBuffer);
            }

            return scales;
        } catch (error) {
            console.error('Multi-resolution analysis creation failed:', error);
            throw error;
        }
    }

    static async analyzeSingleScale(buffer) {
        // Extract various texture features at this scale with enhanced error handling
        const analysisResults = await Promise.allSettled([
            this.extractGaborFeatures(buffer),
            this.extractLawsFeatures(buffer),
            this.extractLBPFeatures(buffer),
            this.extractStatisticalFeatures(buffer),
            this.extractHaralickFeatures(buffer),
            this.extractWaveletFeatures(buffer)
        ]);

        // Process results with appropriate fallbacks
        const [
            gaborFeatures,
            lawsFeatures,
            lbpFeatures,
            statisticalFeatures,
            haralickFeatures,
            waveletFeatures
        ] = analysisResults.map(result => 
            result.status === 'fulfilled' ? result.value : null
        );

        return {
            gabor: gaborFeatures || this.getDefaultGaborFeatures(),
            laws: lawsFeatures || this.getDefaultLawsFeatures(),
            lbp: lbpFeatures || this.getDefaultLBPFeatures(),
            statistical: statisticalFeatures || this.getDefaultStatisticalFeatures(),
            haralick: haralickFeatures || this.getDefaultHaralickFeatures(),
            wavelet: waveletFeatures || this.getDefaultWaveletFeatures(),
            quality: this.assessFeatureQuality({
                gaborFeatures,
                lawsFeatures,
                lbpFeatures,
                statisticalFeatures,
                haralickFeatures,
                waveletFeatures
            })
        };
    }

    static async extractGaborFeatures(buffer) {
        try {
            const features = [];
            const { data, info } = await sharp(buffer)
                .grayscale()
                .raw()
                .toBuffer({ resolveWithObject: true });

            // Create Gabor filter bank
            for (const theta of this.GABOR_ORIENTATIONS) {
                for (const frequency of this.GABOR_FREQUENCIES) {
                    // Create and normalize Gabor kernel
                    const kernel = this.createGaborKernel(theta, frequency);
                    
                    // Apply filter and analyze response
                    const response = await this.applyGaborFilter(data, info.width, info.height, kernel);
                    const responseAnalysis = this.analyzeGaborResponse(response);

                    features.push({
                        orientation: theta,
                        frequency,
                        response: responseAnalysis
                    });
                }
            }

            return {
                responses: features,
                summary: this.summarizeGaborFeatures(features)
            };
        } catch (error) {
            console.error('Gabor feature extraction failed:', error);
            throw error;
        }
    }

    static createGaborKernel(theta, frequency) {
        const size = Math.ceil(6 * this.GABOR_SIGMA);
        const kernel = new Array(size * size);
        const center = Math.floor(size / 2);
        const thetaRad = (theta * Math.PI) / 180;
        const cosTheta = Math.cos(thetaRad);
        const sinTheta = Math.sin(thetaRad);

        let sum = 0;
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const xr = (x - center) * cosTheta + (y - center) * sinTheta;
                const yr = -(x - center) * sinTheta + (y - center) * cosTheta;

                const gaussian = Math.exp(
                    -(xr * xr + this.GABOR_GAMMA * this.GABOR_GAMMA * yr * yr) /
                    (2 * this.GABOR_SIGMA * this.GABOR_SIGMA)
                );

                const oscillation = Math.cos(2 * Math.PI * frequency * xr);
                const value = gaussian * oscillation;
                
                kernel[y * size + x] = value;
                sum += Math.abs(value);
            }
        }

        // Normalize kernel
        for (let i = 0; i < kernel.length; i++) {
            kernel[i] /= sum;
        }

        return kernel;
    }

    static async applyGaborFilter(data, width, height, kernel) {
        const kernelSize = Math.sqrt(kernel.length);
        const halfKernel = Math.floor(kernelSize / 2);
        const response = new Float32Array(width * height);

        // Apply convolution with edge handling
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                let sum = 0;

                for (let ky = -halfKernel; ky <= halfKernel; ky++) {
                    for (let kx = -halfKernel; kx <= halfKernel; kx++) {
                        const ix = x + kx;
                        const iy = y + ky;

                        if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                            const pixelValue = data[iy * width + ix];
                            const kernelValue = kernel[(ky + halfKernel) * kernelSize + (kx + halfKernel)];
                            sum += pixelValue * kernelValue;
                        }
                    }
                }

                response[y * width + x] = sum;
            }
        }

        return response;
    }

    static analyzeGaborResponse(response) {
        // Calculate comprehensive response statistics
        const magnitude = response.map(x => Math.abs(x));
        const energy = this.calculateResponseEnergy(magnitude);
        const entropy = this.calculateResponseEntropy(magnitude);
        
        return {
            mean: mathUtils.calculateMean(magnitude),
            std: Math.sqrt(mathUtils.calculateVariance(magnitude)),
            energy,
            entropy,
            moments: this.calculateResponseMoments(magnitude),
            distribution: this.analyzeResponseDistribution(magnitude)
        };
    }

    static calculateResponseEnergy(magnitude) {
        return magnitude.reduce((sum, val) => sum + val * val, 0) / magnitude.length;
    }

    static calculateResponseEntropy(magnitude) {
        // Create histogram with adaptive binning
        const histogram = this.createAdaptiveHistogram(magnitude);
        const probabilities = this.normalizeHistogram(histogram);

        return -probabilities.reduce((sum, p) => 
            sum + (p > 0 ? p * Math.log2(p) : 0), 0
        );
    }

    static calculateResponseMoments(magnitude) {
        const mean = mathUtils.calculateMean(magnitude);
        const variance = mathUtils.calculateVariance(magnitude, mean);
        const std = Math.sqrt(variance);

        return {
            skewness: mathUtils.calculateSkewness(magnitude, mean, std),
            kurtosis: mathUtils.calculateKurtosis(magnitude, mean, std),
            uniformity: this.calculateUniformity(magnitude)
        };
    }

    static analyzeResponseDistribution(magnitude) {
        // Enhanced distribution analysis with adaptive binning
        const histogram = this.createAdaptiveHistogram(magnitude);
        const normalized = this.normalizeHistogram(histogram);

        return {
            histogram: normalized,
            peaks: this.findDistributionPeaks(normalized),
            valleys: this.findDistributionValleys(normalized),
            modality: this.determineDistributionModality(normalized)
        };
    }

    static createAdaptiveHistogram(data) {
        // Determine optimal bin count using Freedman-Diaconis rule
        const iqr = this.calculateIQR(data);
        const binWidth = 2 * iqr * Math.pow(data.length, -1/3);
        const binCount = Math.ceil((Math.max(...data) - Math.min(...data)) / binWidth);

        const histogram = new Array(binCount).fill(0);
        const min = Math.min(...data);
        const max = Math.max(...data);
        const range = max - min;

        data.forEach(value => {
            const binIndex = Math.min(
                binCount - 1,
                Math.floor((value - min) / range * binCount)
            );
            histogram[binIndex]++;
        });

        return histogram;
    }

    static calculateIQR(data) {
        const sorted = [...data].sort((a, b) => a - b);
        const q1Index = Math.floor(sorted.length * 0.25);
        const q3Index = Math.floor(sorted.length * 0.75);
        return sorted[q3Index] - sorted[q1Index];
    }

static async extractLawsFeatures(buffer) {
        try {
            // Generate Laws' texture energy masks
            const masks = this.generateLawsMasks();
            const features = [];

            // Get grayscale image data
            const { data, info } = await sharp(buffer)
                .grayscale()
                .raw()
                .toBuffer({ resolveWithObject: true });

            // Apply each mask and analyze response
            for (const mask of masks) {
                const response = await this.applyLawsMask(data, info.width, info.height, mask);
                const energyMap = this.calculateTextureEnergy(response, info.width);
                features.push(this.analyzeLawsResponse(energyMap));
            }

            return {
                responses: features,
                summary: this.summarizeLawsFeatures(features)
            };
        } catch (error) {
            console.error('Laws feature extraction failed:', error);
            throw error;
        }
    }

    static generateLawsMasks() {
        // 1D Laws' filter kernels
        const L5 = [1, 4, 6, 4, 1];        // Level
        const E5 = [-1, -2, 0, 2, 1];      // Edge
        const S5 = [-1, 0, 2, 0, -1];      // Spot
        const R5 = [1, -4, 6, -4, 1];      // Ripple
        const W5 = [-1, 2, 0, -2, 1];      // Wave

        const vectors = [L5, E5, S5, R5, W5];
        const masks = [];

        // Generate 2D masks from all combinations
        for (const v1 of vectors) {
            for (const v2 of vectors) {
                const mask = this.generate2DMask(v1, v2);
                masks.push(this.normalizeMask(mask));
            }
        }

        return masks;
    }

    static generate2DMask(v1, v2) {
        const size = v1.length;
        const mask = new Array(size * size);

        for (let i = 0; i < size; i++) {
            for (let j = 0; j < size; j++) {
                mask[i * size + j] = v1[i] * v2[j];
            }
        }

        return mask;
    }

    static async applyLawsMask(data, width, height, mask) {
        const maskSize = Math.sqrt(mask.length);
        const halfMask = Math.floor(maskSize / 2);
        const response = new Float32Array(width * height);

        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                let sum = 0;

                for (let my = 0; my < maskSize; my++) {
                    for (let mx = 0; mx < maskSize; mx++) {
                        const ix = x + mx - halfMask;
                        const iy = y + my - halfMask;

                        if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                            sum += data[iy * width + ix] * mask[my * maskSize + mx];
                        }
                    }
                }

                response[y * width + x] = sum;
            }
        }

        return response;
    }

    static calculateTextureEnergy(response, width) {
        const windowSize = 15;
        const halfWindow = Math.floor(windowSize / 2);
        const height = response.length / width;
        const energyMap = new Float32Array(response.length);

        for (let y = halfWindow; y < height - halfWindow; y++) {
            for (let x = halfWindow; x < width - halfWindow; x++) {
                let energy = 0;

                for (let wy = -halfWindow; wy <= halfWindow; wy++) {
                    for (let wx = -halfWindow; wx <= halfWindow; wx++) {
                        const val = response[(y + wy) * width + (x + wx)];
                        energy += val * val;
                    }
                }

                energyMap[y * width + x] = energy / (windowSize * windowSize);
            }
        }

        return energyMap;
    }

    static analyzeLawsResponse(energyMap) {
        const statistics = this.calculateResponseStatistics(energyMap);
        const peaks = this.findEnergyPeaks(energyMap);
        const patterns = this.analyzeEnergyPatterns(energyMap);

        return {
            statistics,
            peaks,
            patterns,
            distribution: this.analyzeEnergyDistribution(energyMap)
        };
    }

    static calculateResponseStatistics(response) {
        const values = Array.from(response);
        const mean = mathUtils.calculateMean(values);
        const variance = mathUtils.calculateVariance(values);
        const std = Math.sqrt(variance);

        return {
            mean,
            variance,
            std,
            skewness: mathUtils.calculateSkewness(values, mean, std),
            kurtosis: mathUtils.calculateKurtosis(values, mean, std),
            entropy: this.calculateEntropy(values)
        };
    }

    static async extractLBPFeatures(buffer) {
        try {
            const { data, info } = await sharp(buffer)
                .grayscale()
                .raw()
                .toBuffer({ resolveWithObject: true });

            // Enhanced LBP with rotation invariance and uniform patterns
            const lbpImage = new Uint8Array(info.width * info.height);
            const histogram = new Array(256).fill(0);
            const { width, height } = info;

            // Calculate LBP codes with enhanced edge handling
            for (let y = 1; y < height - 1; y++) {
                for (let x = 1; x < width - 1; x++) {
                    const centerPixel = data[y * width + x];
                    let lbpCode = 0;

                    // Sample 8 neighbors
                    for (let i = 0; i < 8; i++) {
                        const ny = y + Math.floor(i / 3) - 1;
                        const nx = x + (i % 3) - 1;
                        
                        if (nx === x && ny === y) continue;

                        const neighborPixel = data[ny * width + nx];
                        if (neighborPixel >= centerPixel) {
                            lbpCode |= (1 << (7 - i));
                        }
                    }

                    lbpImage[y * width + x] = lbpCode;
                    histogram[lbpCode]++;
                }
            }

            // Calculate rotation-invariant uniform patterns
            const riuLBP = this.calculateRotationInvariantUniformLBP(lbpImage);

            // Enhanced LBP analysis
            return {
                histogram: this.normalizeHistogram(histogram),
                riuHistogram: this.normalizeHistogram(riuLBP.histogram),
                patterns: {
                    uniform: riuLBP.uniformCount / (width * height),
                    nonUniform: riuLBP.nonUniformCount / (width * height),
                    patternTypes: this.analyzeLBPPatterns(lbpImage)
                },
                statistics: this.calculateLBPStatistics(histogram),
                mapping: lbpImage,
                quality: this.assessLBPQuality(histogram, riuLBP)
            };
        } catch (error) {
            console.error('LBP feature extraction failed:', error);
            throw error;
        }
    }

    static calculateRotationInvariantUniformLBP(lbpImage) {
        const riuMap = new Array(256).fill(0);
        const riuHistogram = new Array(10).fill(0);
        let uniformCount = 0;
        let nonUniformCount = 0;

        // Build rotation-invariant uniform pattern mapping
        for (let i = 0; i < 256; i++) {
            let pattern = i;
            let minRotation = pattern;
            let transitions = 0;

            // Count transitions
            for (let j = 0; j < 8; j++) {
                const bit1 = (pattern >> j) & 1;
                const bit2 = (pattern >> ((j + 1) % 8)) & 1;
                if (bit1 !== bit2) transitions++;

                // Find minimum rotation
                const rotated = ((pattern << j) | (pattern >> (8 - j))) & 0xFF;
                minRotation = Math.min(minRotation, rotated);
            }

            if (transitions <= 2) {
                riuMap[i] = this.getBitCount(minRotation);
                uniformCount++;
            } else {
                riuMap[i] = 9; // Non-uniform pattern
                nonUniformCount++;
            }
        }

        // Map LBP codes to rotation-invariant uniform patterns
        const histogram = new Array(10).fill(0);
        lbpImage.forEach(code => histogram[riuMap[code]]++);

        return {
            histogram,
            uniformCount,
            nonUniformCount,
            mapping: riuMap
        };
    }

    static analyzeLBPPatterns(lbpImage) {
        const patterns = new Map();
        const visited = new Set();
        const width = Math.sqrt(lbpImage.length);

        // Find connected pattern regions
        for (let y = 1; y < width - 1; y++) {
            for (let x = 1; x < width - 1; x++) {
                const idx = y * width + x;
                if (!visited.has(idx)) {
                    const pattern = this.extractLBPPattern(lbpImage, x, y, width, visited);
                    if (pattern) {
                        const key = pattern.code;
                        if (!patterns.has(key)) {
                            patterns.set(key, []);
                        }
                        patterns.get(key).push(pattern);
                    }
                }
            }
        }

        return this.analyzeLBPPatternDistribution(patterns);
    }

    static extractLBPPattern(lbpImage, startX, startY, width, visited) {
        const code = lbpImage[startY * width + startX];
        const pixels = [];
        const queue = [[startX, startY]];
        const maxSize = 100; // Limit pattern size

        while (queue.length > 0 && pixels.length < maxSize) {
            const [x, y] = queue.shift();
            const idx = y * width + x;

            if (visited.has(idx)) continue;
            visited.add(idx);

            if (lbpImage[idx] === code) {
                pixels.push([x, y]);

                // Check 4-connected neighbors
                const neighbors = [
                    [x+1, y], [x-1, y],
                    [x, y+1], [x, y-1]
                ];

                for (const [nx, ny] of neighbors) {
                    const nidx = ny * width + nx;
                    if (nx >= 0 && nx < width && ny >= 0 && ny < width && 
                        !visited.has(nidx) && lbpImage[nidx] === code) {
                        queue.push([nx, ny]);
                    }
                }
            }
        }

        if (pixels.length < 5) return null; // Minimum pattern size

        return {
            code,
            pixels,
            size: pixels.length,
            bounds: this.calculatePatternBounds(pixels)
        };
    }

    static calculatePatternBounds(pixels) {
        const xs = pixels.map(([x, _]) => x);
        const ys = pixels.map(([_, y]) => y);

        return {
            minX: Math.min(...xs),
            maxX: Math.max(...xs),
            minY: Math.min(...ys),
            maxY: Math.max(...ys),
            width: Math.max(...xs) - Math.min(...xs) + 1,
            height: Math.max(...ys) - Math.min(...ys) + 1
        };
    }

    static analyzeLBPPatternDistribution(patterns) {
        const analysis = {
            totalPatterns: 0,
            patternsByCode: new Map(),
            sizeDistribution: new Map(),
            complexity: 0
        };

        patterns.forEach((patternList, code) => {
            analysis.totalPatterns += patternList.length;
            analysis.patternsByCode.set(code, patternList.length);

            patternList.forEach(pattern => {
                const sizeKey = Math.floor(Math.log2(pattern.size));
                analysis.sizeDistribution.set(
                    sizeKey,
                    (analysis.sizeDistribution.get(sizeKey) || 0) + 1
                );
            });
        });

        // Calculate pattern complexity
        analysis.complexity = this.calculatePatternComplexity(patterns);

        return analysis;
    }

    static calculatePatternComplexity(patterns) {
        let complexity = 0;
        let totalPatterns = 0;

        patterns.forEach((patternList, code) => {
            const bitCount = this.getBitCount(code);
            const transitions = this.countBitTransitions(code);
            
            patternList.forEach(pattern => {
                complexity += (bitCount * transitions * Math.log(pattern.size));
                totalPatterns++;
            });
        });

        return totalPatterns > 0 ? complexity / totalPatterns : 0;
    }

    static countBitTransitions(n) {
        let transitions = 0;
        let lastBit = n & 1;

        for (let i = 1; i < 8; i++) {
            const bit = (n >> i) & 1;
            if (bit !== lastBit) transitions++;
            lastBit = bit;
        }

        // Check transition between last and first bit
        if (lastBit !== (n & 1)) transitions++;

        return transitions;
    }

static async extractHaralickFeatures(buffer) {
        try {
            const { data, info } = await sharp(buffer)
                .grayscale()
                .raw()
                .toBuffer({ resolveWithObject: true });

            // Generate GLCM matrices for multiple distances and angles
            const glcmMatrices = this.generateGLCMMatrices(data, info.width, info.height);
            
            // Calculate Haralick features for each GLCM
            const features = {};
            for (const [direction, matrix] of Object.entries(glcmMatrices)) {
                features[direction] = {
                    contrast: this.calculateContrast(matrix),
                    correlation: this.calculateCorrelation(matrix),
                    energy: this.calculateEnergy(matrix),
                    homogeneity: this.calculateHomogeneity(matrix),
                    entropy: this.calculateEntropy(matrix),
                    clusterShade: this.calculateClusterShade(matrix),
                    clusterProminence: this.calculateClusterProminence(matrix),
                    maximumProbability: this.calculateMaximumProbability(matrix)
                };
            }

            return {
                features,
                summary: this.summarizeHaralickFeatures(features),
                quality: this.assessHaralickQuality(features)
            };
        } catch (error) {
            console.error('Haralick feature extraction failed:', error);
            throw error;
        }
    }

    static async extractWaveletFeatures(buffer) {
        try {
            const { data, info } = await sharp(buffer)
                .grayscale()
                .raw()
                .toBuffer({ resolveWithObject: true });

            const levels = 3;
            const features = [];

            // Perform wavelet decomposition
            let currentData = Float32Array.from(data);
            let currentWidth = info.width;
            let currentHeight = info.height;

            for (let level = 0; level < levels; level++) {
                const [
                    approximation,
                    horizontalDetails,
                    verticalDetails,
                    diagonalDetails
                ] = this.performWaveletDecomposition(
                    currentData, 
                    currentWidth, 
                    currentHeight
                );

                features.push({
                    level,
                    approximation: this.analyzeWaveletSubband(approximation),
                    horizontal: this.analyzeWaveletSubband(horizontalDetails),
                    vertical: this.analyzeWaveletSubband(verticalDetails),
                    diagonal: this.analyzeWaveletSubband(diagonalDetails)
                });

                currentData = approximation;
                currentWidth = Math.floor(currentWidth / 2);
                currentHeight = Math.floor(currentHeight / 2);
            }

            return {
                features,
                energyDistribution: this.analyzeWaveletEnergyDistribution(features),
                directionalityMeasures: this.analyzeWaveletDirectionality(features),
                scaleProperties: this.analyzeWaveletScaleProperties(features)
            };
        } catch (error) {
            console.error('Wavelet feature extraction failed:', error);
            throw error;
        }
    }

    static async detectTexturePatterns(buffer) {
        try {
            const [
                localPatterns,
                globalPatterns,
                periodicPatterns
            ] = await Promise.all([
                this.detectLocalPatterns(buffer),
                this.detectGlobalPatterns(buffer),
                this.detectPeriodicPatterns(buffer)
            ]);

            return {
                local: localPatterns,
                global: globalPatterns,
                periodic: periodicPatterns,
                relationships: this.analyzePatternRelationships({
                    localPatterns,
                    globalPatterns,
                    periodicPatterns
                }),
                hierarchy: this.buildPatternHierarchy({
                    localPatterns,
                    globalPatterns,
                    periodicPatterns
                })
            };
        } catch (error) {
            console.error('Texture pattern detection failed:', error);
            throw error;
        }
    }

    // Default feature generators
    static getDefaultGaborFeatures() {
        return {
            responses: [],
            summary: {
                meanEnergy: 0,
                meanEntropy: 0,
                directionality: 0,
                scale: 0
            }
        };
    }

    static getDefaultLawsFeatures() {
        return {
            responses: [],
            summary: {
                meanEnergy: 0,
                patternStrength: 0,
                roughness: 0,
                regularity: 0
            }
        };
    }

    static getDefaultLBPFeatures() {
        return {
            histogram: new Array(256).fill(1/256),
            riuHistogram: new Array(10).fill(0.1),
            patterns: {
                uniform: 0,
                nonUniform: 0,
                patternTypes: {}
            },
            statistics: this.getDefaultStatistics()
        };
    }

    static getDefaultHaralickFeatures() {
        return {
            features: {},
            summary: {
                contrast: 0,
                correlation: 0,
                energy: 0,
                homogeneity: 0
            }
        };
    }

    static getDefaultWaveletFeatures() {
        return {
            features: [],
            energyDistribution: {
                horizontal: 0,
                vertical: 0,
                diagonal: 0
            },
            directionalityMeasures: {
                horizontalStrength: 0,
                verticalStrength: 0,
                diagonalStrength: 0
            }
        };
    }

    // Quality assessment methods
    static assessFeatureQuality(features) {
        return {
            completeness: this.calculateFeatureCompleteness(features),
            reliability: this.calculateFeatureReliability(features),
            consistency: this.calculateFeatureConsistency(features),
            confidence: this.calculateFeatureConfidence(features)
        };
    }
}

module.exports = TextureAnalysis;

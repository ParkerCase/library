// mathUtils.js

class MathUtils {
    // Matrix operations
    static createMatrix(data) {
        if (!Array.isArray(data)) return null;
        return Array.isArray(data[0]) ? data : [data];
    }

    static multiplyMatrices(a, b) {
        if (!Array.isArray(a) || !Array.isArray(b) || !Array.isArray(b[0])) return null;
        const result = Array(a.length).fill().map(() => Array(b[0].length).fill(0));
        
        for (let i = 0; i < a.length; i++) {
            for (let j = 0; j < b[0].length; j++) {
                for (let k = 0; k < b.length; k++) {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        
        return result;
    }

    static matrixTranspose(matrix) {
        if (!Array.isArray(matrix) || !Array.isArray(matrix[0])) return null;
        return matrix[0].map((_, colIndex) => matrix.map(row => row[colIndex]));
    }

    // Color space conversions
    static rgbToLab(rgb) {
        // Convert RGB to XYZ
        const r = this.gammaCorrect(rgb.r / 255);
        const g = this.gammaCorrect(rgb.g / 255);
        const b = this.gammaCorrect(rgb.b / 255);

        const x = (r * 0.4124564 + g * 0.3575761 + b * 0.1804375) * 100;
        const y = (r * 0.2126729 + g * 0.7151522 + b * 0.0721750) * 100;
        const z = (r * 0.0193339 + g * 0.1191920 + b * 0.9503041) * 100;

        // Convert XYZ to Lab
        return this.xyzToLab({ x, y, z });
    }

    static labToRgb(lab) {
        // Convert Lab to XYZ
        const xyz = this.labToXyz(lab);
        
        // Convert XYZ to RGB
        let r = xyz.x * 3.2404542 - xyz.y * 1.5371385 - xyz.z * 0.4985314;
        let g = -xyz.x * 0.9692660 + xyz.y * 1.8760108 + xyz.z * 0.0415560;
        let b = xyz.x * 0.0556434 - xyz.y * 0.2040259 + xyz.z * 1.0572252;

        // Gamma correction
        r = this.reverseGammaCorrect(r);
        g = this.reverseGammaCorrect(g);
        b = this.reverseGammaCorrect(b);

        // Convert to 8-bit RGB
        return {
            r: Math.round(this.clamp(r * 255, 0, 255)),
            g: Math.round(this.clamp(g * 255, 0, 255)),
            b: Math.round(this.clamp(b * 255, 0, 255))
        };
    }

    static labToXyz(lab) {
        const y = (lab.l + 16) / 116;
        const x = lab.a / 500 + y;
        const z = y - lab.b / 200;

        const xn = 0.95047;
        const yn = 1.00000;
        const zn = 1.08883;

        return {
            x: xn * this.labTransform(x),
            y: yn * this.labTransform(y),
            z: zn * this.labTransform(z)
        };
    }

    static xyzToLab(xyz) {
        const xn = 95.047;
        const yn = 100.000;
        const zn = 108.883;

        const x = xyz.x / xn;
        const y = xyz.y / yn;
        const z = xyz.z / zn;

        const fx = x > 0.008856 ? Math.pow(x, 1/3) : (7.787 * x) + 16/116;
        const fy = y > 0.008856 ? Math.pow(y, 1/3) : (7.787 * y) + 16/116;
        const fz = z > 0.008856 ? Math.pow(z, 1/3) : (7.787 * z) + 16/116;

        return {
            l: (116 * fy) - 16,
            a: 500 * (fx - fy),
            b: 200 * (fy - fz)
        };
    }

    // Statistical functions
    static calculateMean(array) {
        if (!Array.isArray(array) || array.length === 0) return 0;
        return array.reduce((sum, val) => sum + val, 0) / array.length;
    }

    static calculateVariance(array, mean = null) {
        if (!Array.isArray(array) || array.length === 0) return 0;
        const avg = mean === null ? this.calculateMean(array) : mean;
        return array.reduce((sum, val) => sum + Math.pow(val - avg, 2), 0) / array.length;
    }

    static calculateStandardDeviation(array, mean = null, variance = null) {
        if (!Array.isArray(array) || array.length === 0) return 0;
        const var_ = variance === null ? this.calculateVariance(array, mean) : variance;
        return Math.sqrt(var_);
    }

    static calculateSkewness(array, mean = null, stdDev = null) {
        if (!Array.isArray(array) || array.length === 0) return 0;
        const avg = mean === null ? this.calculateMean(array) : mean;
        const std = stdDev === null ? this.calculateStandardDeviation(array, avg) : stdDev;
        
        if (std === 0) return 0;
        
        return array.reduce((sum, val) => 
            sum + Math.pow((val - avg) / std, 3), 0) / array.length;
    }

    static calculateKurtosis(array, mean = null, stdDev = null) {
        if (!Array.isArray(array) || array.length === 0) return 0;
        const avg = mean === null ? this.calculateMean(array) : mean;
        const std = stdDev === null ? this.calculateStandardDeviation(array, avg) : stdDev;
        
        if (std === 0) return 0;
        
        return array.reduce((sum, val) => 
            sum + Math.pow((val - avg) / std, 4), 0) / array.length - 3;
    }

    // Helper functions
    static clamp(value, min, max) {
        return Math.max(min, Math.min(max, value));
    }

    static gammaCorrect(value) {
        return value > 0.04045 ? Math.pow((value + 0.055) / 1.055, 2.4) : value / 12.92;
    }

    static reverseGammaCorrect(value) {
        return value > 0.0031308 ? 1.055 * Math.pow(value, 1/2.4) - 0.055 : 12.92 * value;
    }

    static labTransform(value) {
        return value > 0.206893 ? Math.pow(value, 3) : (value - 16/116) / 7.787;
    }

    // Distance calculations
    static euclideanDistance(point1, point2) {
        if (!Array.isArray(point1) || !Array.isArray(point2) || point1.length !== point2.length) {
            return null;
        }
        return Math.sqrt(
            point1.reduce((sum, val, idx) => sum + Math.pow(val - point2[idx], 2), 0)
        );
    }

    static deltaE(lab1, lab2) {
        const deltaL = lab1.l - lab2.l;
        const deltaA = lab1.a - lab2.a;
        const deltaB = lab1.b - lab2.b;
        return Math.sqrt(deltaL * deltaL + deltaA * deltaA + deltaB * deltaB);
    }

    // Vector operations
    static dotProduct(vec1, vec2) {
        if (!Array.isArray(vec1) || !Array.isArray(vec2) || vec1.length !== vec2.length) {
            return null;
        }
        return vec1.reduce((sum, val, idx) => sum + val * vec2[idx], 0);
    }

    static vectorMagnitude(vec) {
        if (!Array.isArray(vec)) return null;
        return Math.sqrt(vec.reduce((sum, val) => sum + val * val, 0));
    }

    static normalizeVector(vec) {
        if (!Array.isArray(vec)) return null;
        const magnitude = this.vectorMagnitude(vec);
        if (magnitude === 0) return vec.map(() => 0);
        return vec.map(val => val / magnitude);
    }

    // Convolution operations
    static convolve2D(matrix, kernel) {
        if (!Array.isArray(matrix) || !Array.isArray(kernel)) return null;
        
        const mRows = matrix.length;
        const mCols = matrix[0].length;
        const kRows = kernel.length;
        const kCols = kernel[0].length;
        
        const result = Array(mRows - kRows + 1).fill()
            .map(() => Array(mCols - kCols + 1).fill(0));

        for (let i = 0; i <= mRows - kRows; i++) {
            for (let j = 0; j <= mCols - kCols; j++) {
                let sum = 0;
                for (let k = 0; k < kRows; k++) {
                    for (let l = 0; l < kCols; l++) {
                        sum += matrix[i + k][j + l] * kernel[k][l];
                    }
                }
                result[i][j] = sum;
            }
        }

        return result;
    }
}

module.exports = MathUtils;

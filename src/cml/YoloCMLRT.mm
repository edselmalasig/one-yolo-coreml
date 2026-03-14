#ifdef BUILD_WITH_CML

// 1. OpenCV FIRST — before any ObjC headers pollute the preprocessor
#include "YoloCMLRT.h"

// 2. Undef ObjC boolean macros that clash with OpenCV enums
#ifdef NO
#undef NO
#endif
#ifdef YES
#undef YES
#endif

// 3. Now safe to bring in ObjC/CoreML
#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <CoreVideo/CoreVideo.h>

#include <stdexcept>
#include <cstring>

namespace yolo {

YoloCMLRT::YoloCMLRT(const std::string& model_path) : YoloRuntime("CoreML") {
    @autoreleasepool {
        NSString* path   = [NSString stringWithUTF8String:model_path.c_str()];
        NSURL*    srcURL = [NSURL fileURLWithPath:path];
        NSError*  error  = nil;

        NSLog(@"[YoloCMLRT] Compiling %@ ...", path);
        NSURL* compiledURL = [MLModel compileModelAtURL:srcURL error:&error];
        if (!compiledURL) {
            std::string msg = "YoloCMLRT: compileModelAtURL failed for " + model_path;
            if (error) msg += std::string(" — ") + [[error localizedDescription] UTF8String];
            throw std::runtime_error(msg);
        }

        MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
        config.computeUnits = MLComputeUnitsAll;

        MLModel* model = [MLModel modelWithContentsOfURL:compiledURL
                                           configuration:config
                                                   error:&error];
        if (!model) {
            std::string msg = "YoloCMLRT: modelWithContentsOfURL failed";
            if (error) msg += std::string(" — ") + [[error localizedDescription] UTF8String];
            throw std::runtime_error(msg);
        }

        // Inspect input to decide feed mode
        MLModelDescription* desc = model.modelDescription;
        NSString* inputKey = desc.inputDescriptionsByName.allKeys.firstObject;
        MLFeatureDescription* inputDesc = desc.inputDescriptionsByName[inputKey];
        __inputIsImage = (inputDesc.type == MLFeatureTypeImage);
        NSLog(@"[YoloCMLRT] Input '%@' type: %@", inputKey,
              __inputIsImage ? @"Image(CVPixelBuffer)" : @"MultiArray");

        [model retain];
        __model = (void*)model;
        NSLog(@"[YoloCMLRT] Model loaded OK");
    }
}

YoloCMLRT::~YoloCMLRT() {
    if (__model) {
        [(MLModel*)__model release];
        __model = nullptr;
    }
}

// Build a CVPixelBufferRef from a pre-normalised float blob [1,3,H,W] NCHW.
// CoreML Image inputs expect BGRA or 32ARGB pixel buffers; we use 32BGRA and
// fill it from the float planes after scaling back to uint8.
static CVPixelBufferRef blobToPixelBuffer(const cv::Mat& blob) {
    // blob: [1, 3, H, W] float32, values in [0,1]
    const int H = blob.size[2];
    const int W = blob.size[3];

    // Reconstruct an interleaved BGR uint8 mat from the NCHW blob
    // blob channel order matches cfg.rgb (default true → RGB planes)
    std::vector<cv::Mat> planes(3);
    for (int c = 0; c < 3; ++c) {
        // Each plane is a sub-matrix view into the blob
        int sz[2] = {H, W};
        planes[c] = cv::Mat(2, sz, CV_32F,
                            const_cast<float*>(blob.ptr<float>()) + c * H * W);
    }
    cv::Mat rgbFloat, rgb8, bgra;
    cv::merge(planes, rgbFloat);               // HxWx3 float32 RGB [0,1]
    rgbFloat.convertTo(rgb8, CV_8UC3, 255.0);  // HxWx3 uint8 RGB
    cv::cvtColor(rgb8, bgra, cv::COLOR_RGB2BGRA); // HxWx4 BGRA

    CVPixelBufferRef pb = nullptr;
    CVReturn ret = CVPixelBufferCreate(
        kCFAllocatorDefault, W, H,
        kCVPixelFormatType_32BGRA,
        nullptr, &pb);
    if (ret != kCVReturnSuccess || !pb)
        throw std::runtime_error("YoloCMLRT: CVPixelBufferCreate failed");

    CVPixelBufferLockBaseAddress(pb, 0);
    uint8_t* dst = (uint8_t*)CVPixelBufferGetBaseAddress(pb);
    size_t   rowBytes = CVPixelBufferGetBytesPerRow(pb);
    for (int y = 0; y < H; ++y)
        std::memcpy(dst + y * rowBytes, bgra.ptr(y), W * 4);
    CVPixelBufferUnlockBaseAddress(pb, 0);

    return pb;
}

std::vector<cv::Mat> YoloCMLRT::inference(const cv::Mat& blob) {
    CV_Assert(blob.dims == 4 && blob.type() == CV_32F);

    std::vector<cv::Mat> outputs;

    @autoreleasepool {
        const int N = blob.size[0];
        const int C = blob.size[1];
        const int H = blob.size[2];
        const int W = blob.size[3];
        (void)C;

        MLModel*  model    = (MLModel*)__model;
        NSError*  error    = nil;
        NSString* inputKey =
            model.modelDescription.inputDescriptionsByName.allKeys.firstObject;

        MLFeatureValue* fv = nil;

        if (__inputIsImage) {
            // Feed a CVPixelBuffer
            CVPixelBufferRef pb = blobToPixelBuffer(blob);
            fv = [MLFeatureValue featureValueWithPixelBuffer:pb];
            CVPixelBufferRelease(pb);
        } else {
            // Feed a MultiArray
            NSArray<NSNumber*>* shape = @[@(N), @(C), @(H), @(W)];
            MLMultiArray* inputArray =
                [[MLMultiArray alloc] initWithShape:shape
                                           dataType:MLMultiArrayDataTypeFloat32
                                              error:&error];
            if (!inputArray)
                throw std::runtime_error("YoloCMLRT: failed to allocate MLMultiArray");
            const float* src = blob.ptr<float>();
            float*       dst = (float*)[inputArray dataPointer];
            std::memcpy(dst, src, (size_t)N * C * H * W * sizeof(float));
            fv = [MLFeatureValue featureValueWithMultiArray:inputArray];
        }

        MLDictionaryFeatureProvider* provider =
            [[MLDictionaryFeatureProvider alloc]
                initWithDictionary:@{inputKey: fv}
                             error:&error];
        if (!provider)
            throw std::runtime_error("YoloCMLRT: failed to create feature provider");

        id<MLFeatureProvider> result =
            [model predictionFromFeatures:provider error:&error];
        if (!result) {
            std::string msg = "YoloCMLRT: prediction failed";
            if (error) msg += std::string(" — ") + [[error localizedDescription] UTF8String];
            throw std::runtime_error(msg);
        }

        NSDictionary* outDescs = model.modelDescription.outputDescriptionsByName;
        for (NSString* key in outDescs.allKeys) {
            MLMultiArray* oarr = [[result featureValueForName:key] multiArrayValue];
            if (!oarr) continue;

            std::vector<int> sizes;
            for (NSNumber* dim in oarr.shape)
                sizes.push_back(dim.intValue);

            int total = 1;
            for (int d : sizes) total *= d;

            cv::Mat out((int)sizes.size(), sizes.data(), CV_32F);
            std::memcpy(out.ptr<float>(),
                        (const float*)[oarr dataPointer],
                        total * sizeof(float));
            outputs.push_back(out);
        }
    }

    return outputs;
}

} // namespace yolo

#endif // BUILD_WITH_CML
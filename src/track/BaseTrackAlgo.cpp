#include "track/BaseTrackAlgo.h"

namespace yolo {
    BaseTrackAlgo::BaseTrackAlgo(const YoloTrackConfig& cfg) {
        _cfg = cfg;
    }
    
    BaseTrackAlgo::~BaseTrackAlgo() {

    }
}
#include "modules/common/util/colormap.h"

#include <vector>

namespace roadstar {
namespace common {
namespace util {

std::unordered_map<ColorName, Color, EnumHash> Color::map_ = {
    {ColorName::Red, 0xff0000_rgb},
    {ColorName::Green, 0x00ff00_rgb},
    {ColorName::Blue, 0x0000ff_rgb},
    {ColorName::Black, 0x000000_rgb},
    {ColorName::White, 0xffffff_rgb},
    {ColorName::GhostWhite, 0xf8f8ff_rgb},
    {ColorName::Yellow, 0xffff00_rgb},
    {ColorName::Cyan, 0x00ffff_rgb},
    {ColorName::Orange, 0xff6100_rgb},
    {ColorName::Purple, 0xa020f0_rgb},
    {ColorName::RoyalBlue, 0x4169e1_rgb},
    {ColorName::GreenEarth, 0x385e0f_rgb},
    {ColorName::Chocolate, 0xd2691e_rgb},
    {ColorName::IndianRed, 0xb0171f_rgb},
    {ColorName::DeepRed, 0xff00ff_rgb},
    {ColorName::Flaxen, 0xfafee6_rgb},
    {ColorName::PeakGreen, 0x00ff7f_rgb},
    {ColorName::SquidInkBrown, 0x5e2612_rgb},
    {ColorName::Violet, 0x8a2be6_rgb},
    {ColorName::MeiRed, 0xdda0dd_rgb},
    {ColorName::PeacockBlue, 0x33a1c9_rgb},
    {ColorName::TurkishJade, 0x00c78c_rgb},
    {ColorName::SlateGray, 0x708069_rgb},
    {ColorName::Gray, 0xc0c0c0_rgb},
};

}  // namespace util
}  // namespace common
}  // namespace roadstar

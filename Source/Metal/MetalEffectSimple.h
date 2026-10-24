// Copyright (c) 2026 RTGL1 Metal backend contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

// Included by EffectSimple.h, if RG_USE_METAL:
// EffectSimple of the Vulkan backend, without Vulkan objects -- only the push constants and
// the transition state, so that the effects of EffectSimple_Instances.h are shared.
// The dispatch is done by MetalPathTracer.

#include "RTGL1/RTGL1.h"

#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <string_view>

namespace RTGL1
{

struct CommonnlyUsedEffectArguments
{
    const uint32_t width, height;
    const float    currentTime;
};

struct EmptyPushConst
{
};

template< typename PushConst, StringLiteral ShaderName >
struct EffectSimple
{
    EffectSimple() : push{}, isCurrentlyActive{ false } {}
    virtual ~EffectSimple() = default;

    EffectSimple( const EffectSimple& other )            = delete;
    EffectSimple& operator=( const EffectSimple& other ) = delete;

protected:
    bool SetupNull()
    {
        isCurrentlyActive = false;
        return false;
    }

    // the same as the Vulkan EffectSimple::Setup
    bool Setup( const CommonnlyUsedEffectArguments& args,
                bool                                isActive,
                float                               transitionDurationIn,
                float                               transitionDurationOut )
    {
        bool wasActivePreviously = isCurrentlyActive;
        isCurrentlyActive        = isActive;

        // if to start
        if( !wasActivePreviously && isCurrentlyActive )
        {
            push.transitionType      = 0;
            push.transitionBeginTime = args.currentTime;
            push.transitionDuration  = transitionDurationIn;
        }
        // if to end
        else if( wasActivePreviously && !isCurrentlyActive )
        {
            push.transitionType      = 1;
            push.transitionBeginTime = args.currentTime;
            push.transitionDuration  = transitionDurationOut;
        }

        return isCurrentlyActive ||
               ( push.transitionType == 1 &&
                 args.currentTime - push.transitionBeginTime <= push.transitionDuration );
    }

public:
    // "EffectWaves" -> "EfWaves", the compiled shader's name
    static std::string GetShaderName()
    {
        constexpr auto prefix = std::string_view{ "Effect" };

        auto name = std::string_view{ ShaderName.value };
        if( name.starts_with( prefix ) )
        {
            return std::string{ "Ef" } + std::string{ name.substr( prefix.size() ) };
        }
        return std::string{ name };
    }

    // GLSL's push_constant block of EfSimple.inl
    std::span< const uint8_t > GetPushBytes() const
    {
        return { reinterpret_cast< const uint8_t* >( &push ), sizeof( push ) };
    }

    // Can be changed by child classes
    PushConst& GetPush() { return push.custom; }

protected:
    struct
    {
        uint32_t  transitionType; // 0 - in, 1 - out
        float     transitionBeginTime;
        float     transitionDuration;
        PushConst custom;
    } push;

private:
    bool isCurrentlyActive;
};

}

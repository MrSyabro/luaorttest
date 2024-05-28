local Ort = require "luaort"
local png = require "luapng"

F = 2

local Env = Ort.CreateEnv()
local SessionOptions = Ort.CreateSessionOptions()
local Session = Env:CreateSession("quantized.onnx", SessionOptions)

local imagedata, dh, dw = png.read("butterfly.png")
local imagetensor = Ort.CreateValue({ 1, 3, dh, dw }, "FLOAT",  imagedata)

local outputvalues = Session:Run {
	pixel_values = imagetensor
}

local outputImage = outputvalues.reconstruction:GetData()
png.write(outputImage, dh*F, dw*F, "out.png")
local Ort = require "luaort"
local png = require "luapng"
local vec = require "vec"

F = 2

local Env = Ort.CreateEnv()
local SessionOptions = Ort.CreateSessionOptions()
local Session = Env:CreateSession("quantized.onnx", SessionOptions)

local imagedata, dh, dw = png.read("butterfly.png")
imagedata = vec.fromfloatb(imagedata)
local imagetensor = Ort.CreateValue({ 1, 3, dh, dw }, "FLOAT",  imagedata)

local outputvalues = Session:Run {
	pixel_values = imagetensor
}

local outputImage = outputvalues.reconstruction:GetData()
png.write(outputImage, dh*F, dw*F, "out.png")
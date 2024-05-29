local Ort = require "luaort"
local png = require "luapng"
local vec = require "vec"

local Env = Ort.CreateEnv()
local SessionOptions = Ort.CreateSessionOptions()
local Session = Env:CreateSession("candy.onnx", SessionOptions)

local imagedata, dh, dw = png.read("in.png")
imagedata = vec.fromfloatb(imagedata)
local imagetensor = Ort.CreateValue({ 1, 3, dh, dw }, "FLOAT",  imagedata)

local outputvalues = Session:Run {
	inputImage = imagetensor
}

local outputImage = outputvalues.outputImage:GetData()
png.write(outputImage, dh, dw, "out.png")
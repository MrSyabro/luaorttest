local Ort = require "luaort"
local png = require "luapng"
local vec = require "vec"

print("Reading girl.png")
local imagedata, dh, dw = png.read("girl.png")
imagedata = vec.fromfloatb(imagedata):div(255)
local imagetensor = Ort.CreateValue({ 1, 3, dh, dw }, "FLOAT",  imagedata)
print("Size:", #imagedata, dw, dh)

print("Reading mask.png")
local maskdata, mh, mw = png.read_grayscale("mask.png")
maskdata = vec.fromfloatb(maskdata):div(255)
local masktensor = Ort.CreateValue({ 1, 1, mh, mw }, "FLOAT",  maskdata)
print("Size:", #maskdata, mw, mh)

local Env = Ort.CreateEnv()
local SessionOptions = Ort.CreateSessionOptions()
print "Loading model"
local Session = Env:CreateSession("lama_fp32.onnx", SessionOptions)

print "Run"
local outputvalues = Session:Run {
	image = imagetensor,
	mask = masktensor,
}

local outputImage = outputvalues.output:GetData()
png.write(outputImage, dh, dw, "out.png")
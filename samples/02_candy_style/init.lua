local Ort = require "luaort"
local png = require "luapng"

print "Loading model"
local Env = Ort.CreateEnv()
local SessionOptions = Ort.CreateSessionOptions()
local Session = Env:CreateSession("candy.onnx", SessionOptions)

local imagedata = png.read("in.png")
local imagetensor = Ort.CreateValue({ 1, 3, imagedata.height, imagedata.width }, "FLOAT",  imagedata)

print "Run"
local outputvalues = Session:Run {
	inputImage = imagetensor
}

print "saving"
local outputImage = outputvalues.outputImage:GetData()
outputImage.height = imagedata.height
outputImage.width = imagedata.width
png.write(outputImage, "out.png")
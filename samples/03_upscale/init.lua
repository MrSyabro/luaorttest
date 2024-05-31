local Ort = require "luaort"
local png = require "luapng"

F = 2

print "Loading model"
local Env = Ort.CreateEnv()
local SessionOptions = Ort.CreateSessionOptions()
local Session = Env:CreateSession("quantized.onnx", SessionOptions)

local imagedata = png.read("butterfly.png")
local imagetensor = Ort.CreateValue({ 1, 3, imagedata.height, imagedata.width }, "FLOAT",  imagedata)

print "Run"
local outputvalues = Session:Run {
	pixel_values = imagetensor
}

local outputImage = outputvalues.reconstruction:GetData()
outputImage.height = imagedata.height * F
outputImage.width = imagedata.width * F
png.write(outputImage, "out.png")
print "Finish"
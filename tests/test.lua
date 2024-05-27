local Ort = require "luaort" --[[@as Ort]]
local png = require "luapng" --[[@as PNGUtils]]

local dw, dh, d = png.read("images/in.png")
d = png.hwc2chw(d)

local Env = Ort.CreateEnv()

local SessionOptions = Ort.CreateSessionOptions()

print("Loading model")
local Session = Env:CreateSession("models/candy.onnx", SessionOptions)

local inputs = Session:GetInputs()
local outputs = Session:GetOutputs()

local inputvalue = Ort.CreateValue({ 1, 3, dw, dh }, "FLOAT", d)

print("Running")
local outputvalues = Session:Run(inputs, {inputvalue}, outputs)

local d = outputvalues[1]:GetData()
d = png.chw2hwc(d)
png.write(d, dw, dh, "out.png")
os.execute("start out.png")
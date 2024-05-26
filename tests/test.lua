local Ort = require "luaort" --[[@as Ort]]
local png = require "luapng" --[[@as PNGUtils]]

local dw, dh, d = png.read("girl.png")
d = png.hwc2chw(d)
print("Data size", #d)

local maskw, maskh, maskdata = png.read_grayscale("mask.png")
maskdata = png.hwc2chw(maskdata)
print("Mask2 size", #maskdata)

local Env = Ort.CreateEnv()

local SessionOptions = Ort.CreateSessionOptions()

print("Loading model")
local Session = Env:CreateSession("lama_fp32.onnx", SessionOptions)

local inputs = Session:GetInputs()
print("Inputs:")
for i, name in ipairs(inputs) do
	local datatype, dims = Session:GetInputType(i)
	io.write(("%d %s(%s): {"):format(i, name, datatype))
	for i, dim in ipairs(dims) do
		io.write(tostring(dim))
		if i ~= #dims then io.write(", ") end
	end
	io.write("}\n")
end

local outputs = Session:GetOutputs()
print("Outputs:")
for i, name in ipairs(outputs) do
	local datatype, dims = Session:GetOutputType(i)
	io.write(("%d %s(%s): {"):format(i, name, datatype))
	for i, dim in ipairs(dims) do
		io.write(tostring(dim))
		if i ~= #dims then io.write(", ") end
	end
	io.write("}\n")
end

--print("PAUSE"); io.read("l")

local inputvalue = Ort.CreateValue({ 1, 3, dw, dh }, "FLOAT", d)
local maskvalue = Ort.CreateValue({1, 1, maskw, maskh}, "FLOAT", maskdata)

print("Running")
local outputvalues = Session:Run(inputs, {inputvalue, maskvalue}, outputs)

local d = outputvalues[1]:GetData()
print("size of outdata", #d)

d = png.chw2hwc(d)
print("data2 size", #d)

png.write(d, dw, dh, "out.png")

os.execute("start out.png")
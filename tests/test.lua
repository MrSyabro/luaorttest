local Ort = require "luaort" --[[@as Ort]]
local png = require "luapng" --[[@as PNGUtils]]

local w, h, d = png.read("in.png")
print("Png size", w, h)
print("string size", #d)
d = png.hwc2chw(d)

local Env = Ort.CreateEnv()

local SessionOptions = Ort.CreateSessionOptions()

local Session = Env:CreateSession("candy.onnx", SessionOptions)

local inputs = Session:GetInputs()
print("Inputs:")
for i, name in ipairs(inputs) do
	local datatype, dims = Session:GetOutputType(1)
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
	local datatype, dims = Session:GetOutputType(1)
	io.write(("%d %s(%s): {"):format(i, name, datatype))
	for i, dim in ipairs(dims) do
		io.write(tostring(dim))
		if i ~= #dims then io.write(", ") end
	end
	io.write("}\n")
end

local inputvalue = Ort.CreateValue({ 1, 3, w, h }, "FLOAT", d)
local outputvalues = Session:Run(inputs, {inputvalue}, outputs)

local d = outputvalues[1]:GetData()
print("size of outdata", #d)

d = png.chw2hwc(d)
print("data2 size", #d)

png.write(d, w, h, "out.png")

os.execute("start out.png")
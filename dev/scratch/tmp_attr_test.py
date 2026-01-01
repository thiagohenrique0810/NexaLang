from llvmlite import ir
m=ir.Module(name='t')
m.triple='spirv64-unknown-vulkan'
fnty=ir.FunctionType(ir.VoidType(), [])
f=ir.Function(m,fnty,name='k')
try:
    f.attributes.add('hlsl.shader','compute')
except Exception as e:
    print('tuple-add error:',e)

for s in [
    '"hlsl.shader"="compute"',
    'hlsl.shader="compute"',
    'hlsl.shader=compute',
    '"hlsl.numthreads"="8,1,1"',
]:
    m2=ir.Module(name='t2')
    m2.triple=m.triple
    f2=ir.Function(m2,fnty,name='k')
    try:
        f2.attributes.add(s)
        print('OK:',s)
        # print only attributes line(s)
        for line in str(m2).splitlines():
            if line.startswith('attributes #'):
                print(line)
    except Exception as e:
        print('FAIL:',s,e)

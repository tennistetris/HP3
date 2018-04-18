NVCC = nvcc

Assignment_3_14ME10078: headers.h   
	$(NVCC) -I ./ kernel.cu host.cu -o Assignment_3_14ME10078
run:
	./Assignment_3_14ME10078 < input.txt
clean:
	rm -rf Assignment_3_14ME10078

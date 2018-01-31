NVCC = nvcc

Assignment_1_14ME10078: headers.h kernel_sincosadd.cu kernel_log.cu kernel_sqrt.cu  
	$(NVCC) -I ./ kernel_sincosadd.cu kernel_log.cu kernel_sqrt.cu host.cu -o Assignment_1_14ME10078

      
run:
	./Assignment_1_14ME10078

clean:
	rm -rf Assignment_1_14ME10078

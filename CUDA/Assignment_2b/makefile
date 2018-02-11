NVCC = nvcc

Assignment_2_14ME10078: headers.h Conv1D.cu Conv2D.cu
	$(NVCC) -I ./ host.cu Conv1D.cu Conv2D.cu -o Assignment_2b_14ME10078
run:
	./Assignment_2b_14ME10078
clean:
	rm -rf Assignment_2b_14ME10078

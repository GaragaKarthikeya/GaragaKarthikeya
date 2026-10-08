
# Garaga Karthikeya

**Computer Architecture · FPGA · AI Accelerators · Memory Systems**

Third-year Integrated M.Tech student in Electronics and Communication Engineering at **IIIT Bangalore**. Research Intern at the IHS Lab, working with Prof. Madhav Rao.

I'm interested in how computation is organized across the hardware stack, particularly where architecture, memory systems, and physical implementation meet.

I enjoy taking ideas from papers to working implementations, questioning established abstractions, and measuring what actually happens beyond simulation.

## Research & Projects

### [INSITU — FPGA Attention Accelerator](https://github.com/GaragaKarthikeya/INSITU)
`FPGA` `Verilog` `LLM Inference` `Hardware Validation`

An FPGA accelerator that computes attention directly over compressed KV-cache representations, without reconstructing the original keys and values.

- Validated all **16 Llama-3.2-1B attention layers** on a Xilinx ZCU104, with bit-exact agreement against NumPy over **16,384 decode steps**.
- Built automated hardware bring-up and verification tooling using Python, UDP, UART, and JTAG.
- Achieved **4.92× KV-cache reduction** at a **1.17× perplexity cost**, meeting **250 MHz** timing.

### [POLARIS — Online Memory Controller Tuning](https://github.com/GaragaKarthikeya/POLARIS)
`Computer Architecture` `PIM` `LPDDR5` `Ramulator 2.0`

A runtime agent for per-kernel adaptation of concurrent PIM/CPU memory controllers under CPU slowdown constraints.

- Uses **Thompson sampling** and per-application learning to tune five memory-controller parameters.
- Achieves **10–15% higher PIM throughput** than the best fixed configuration.
- Evaluated through **72,900 simulations** across three LLMs and twelve CPU workloads, with reproducibility scripts and experimental data available in the repository.

### [8-bit Breadboard CPU](https://github.com/GaragaKarthikeya/Report-for-Breadboard-CPU-Project)
`Digital Logic` `Computer Organization` `Embedded Systems`

A functional 8-bit processor built from discrete TTL logic rather than a prebuilt processor core.

- Designed a custom **12-instruction ISA**, datapath, registers, and control unit.
- Coordinated three Arduinos over I²C to drive 20 control signals.
- Integrated and debugged the processor on physical hardware.

## Technical Background

| Domain | Technologies |
|---|---|
| Hardware Design | Verilog, SystemVerilog, Vivado, FPGA |
| Programming | Python, C/C++, Bash, Linux |
| Architecture & Memory | DDR, LPDDR5, PIM, Ramulator 2.0 |
| Interfaces & Debug | AXI, UART, JTAG, UDP, I²C |
| Machine Learning | PyTorch, Reinforcement Learning, GNNs |

## Currently Exploring

- Processing-in-memory and memory-centric architectures
- Domain-specific accelerators and hardware/software co-design
- FPGA architecture and physical design
- Digital and analog VLSI, from circuits to systems

## Beyond Research

I like peeling back abstraction layers until I understand what makes a system work. Outside engineering, I play guitar.

## Connect

[Email](mailto:garaga.karthikeya@iiitb.ac.in) · [LinkedIn](https://linkedin.com/in/karthikeya-garaga)

---

*Building things, and figuring out what's worth building next.*

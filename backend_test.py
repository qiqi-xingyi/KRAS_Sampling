# --*-- conding:utf-8 --*--
# @time:12/11/25 00:08
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:backend_test.py


import sys
from qiskit_ibm_runtime import QiskitRuntimeService


def main():
    print("=== IBM Quantum connection test ===")

    try:
        service = QiskitRuntimeService()
        print("[OK] QiskitRuntimeService initialized from local account.")
    except Exception as exc:
        print("[ERROR] Failed to initialize QiskitRuntimeService from local account.")
        print("        Make sure you have run QiskitRuntimeService.save_account(...)")
        print("        Exception message:")
        print(f"        {exc}")
        sys.exit(1)


    try:
        instance = getattr(service, "instance", None)
        if instance:
            print(f"[INFO] Using instance: {instance}")
        else:
            print("[INFO] Instance information not available (attribute 'instance' missing).")
    except Exception:
        print("[INFO] Unable to query service.instance (not critical).")


    try:
        backends = service.backends()
        if not backends:
            print("[WARN] No backends visible for this account/instance.")
        else:
            print(f"[OK] Found {len(backends)} backends. Showing first 5:")
            for backend in backends[:5]:
                try:
                    bname = backend.name
                except AttributeError:
                    bname = getattr(backend, "backend_name", "<unknown>")

                try:
                    status = backend.status()
                    operational = getattr(status, "operational", None)
                    pending_jobs = getattr(status, "pending_jobs", None)
                except Exception:
                    operational = None
                    pending_jobs = None

                try:
                    num_qubits = getattr(backend.configuration(), "num_qubits", None)
                except Exception:
                    num_qubits = None

                print(f"  - {bname} | qubits={num_qubits} | "
                      f"operational={operational} | pending_jobs={pending_jobs}")
    except Exception as exc:
        print("[ERROR] Failed to list backends from service.backends().")
        print(f"        {exc}")
        sys.exit(1)


    if len(sys.argv) > 1:
        backend_name = sys.argv[1]
        print(f"\n=== Detailed info for backend: {backend_name} ===")
        try:
            backend = service.backend(backend_name)
        except Exception as exc:
            print(f"[ERROR] Could not load backend '{backend_name}'.")
            print(f"        {exc}")
            sys.exit(1)


        try:
            config = backend.configuration()
        except Exception:
            config = None

        try:
            status = backend.status()
        except Exception:
            status = None

        print(f"Name: {getattr(backend, 'name', backend_name)}")
        if config is not None:
            print(f"  num_qubits: {getattr(config, 'num_qubits', 'N/A')}")
            print(f"  simulator: {getattr(config, 'simulator', 'N/A')}")
            print(f"  backend_version: {getattr(config, 'backend_version', 'N/A')}")
        else:
            print("  [WARN] Unable to get backend.configuration()")

        if status is not None:
            print(f"  operational: {getattr(status, 'operational', 'N/A')}")
            print(f"  pending_jobs: {getattr(status, 'pending_jobs', 'N/A')}")
            print(f"  status_msg: {getattr(status, 'status_msg', 'N/A')}")
        else:
            print("  [WARN] Unable to get backend.status()")

    print("\n=== Test completed. ===")


if __name__ == "__main__":
    main()

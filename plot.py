def plot_comparison(results):
    import matplotlib.pyplot as plt

    for method, aucs in results.items():
        plt.plot(aucs, label=method)

    plt.xlabel("Rounds")
    plt.ylabel("AUC")
    plt.legend()
    plt.title("Method Comparison")

    plt.show()
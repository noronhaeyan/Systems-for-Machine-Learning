from functools import cache
import matplotlib.pyplot as plt


def main():
    # Your main code goes here
    print("Hello, World!")


@cache
def temporal(k):
    if k == 0:
        return 0

    if k == 1:
        return 12.5

    output = k + 12.5

    for i in range(k):
        output += 0.25 * temporal(k - i - 1) * (0.75**i)

    return output


def temporal2(k):
    output = k * (0.75**k)
    for i in range(k - 1):
        output += i * 0.25 * (0.75**i)
    return output


if __name__ == "__main__":
    input = []
    output = []
    for i in range(1, 20):
        # print("for token length: ", i, ", time taken is: ", temporal(i))
        input.append(i)
        # output.append(12.5 * i / temporal(i))
        output.append(temporal2(i) * 12.5 / (i + 12.5))

    plt.plot(input, output)
    plt.xlabel("Token Length")
    plt.ylabel("Speed Up")
    plt.title("Speed Up vs Token Length")
    plt.show()

    for i in range(len(output)):
        print(f"Token Length: {input[i]}, Speed Up: {output[i]}")

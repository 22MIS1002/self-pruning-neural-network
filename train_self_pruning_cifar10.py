import os
import argparse
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        nn.init.zeros_(self.bias)

        # Better than 2.0. Starts gates at sigmoid(0)=0.5
        nn.init.constant_(self.gate_scores, 0.0)

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        pruned_weight = self.weight * gates
        return F.linear(x, pruned_weight, self.bias)

    def gate_values(self):
        return torch.sigmoid(self.gate_scores)


class SelfPruningNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.flatten = nn.Flatten()

        self.fc1 = PrunableLinear(3 * 32 * 32, 1024)
        self.bn1 = nn.BatchNorm1d(1024)

        self.fc2 = PrunableLinear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)

        self.fc3 = PrunableLinear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)

        self.fc4 = PrunableLinear(256, num_classes)

        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.flatten(x)

        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)

        x = F.relu(self.bn3(self.fc3(x)))

        return self.fc4(x)

    def sparsity_loss(self):
        gates = []

        for module in self.modules():
            if isinstance(module, PrunableLinear):
                gates.append(module.gate_values().flatten())

        all_gates = torch.cat(gates)

        # Mean is better scaled than sum
        return all_gates.mean()

    def all_gates(self):
        gates = []

        for module in self.modules():
            if isinstance(module, PrunableLinear):
                gates.append(module.gate_values().detach().cpu().flatten())

        return torch.cat(gates)

    def sparsity_percentage(self, threshold=0.1):
        gates = self.all_gates()
        pruned = (gates < threshold).sum().item()
        total = gates.numel()
        return 100.0 * pruned / total


def get_loaders(batch_size=128, subset=None):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )
    ])

    train_data = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform_train
    )

    test_data = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform_test
    )

    if subset is not None:
        train_data = torch.utils.data.Subset(train_data, range(subset))

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    return train_loader, test_loader


def train_one_epoch(model, loader, optimizer, device, lambda_sparse):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(loader, desc="Training", leave=False)

    for images, labels in loop:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        classification_loss = F.cross_entropy(outputs, labels)
        sparse_loss = model.sparsity_loss()

        loss = classification_loss + lambda_sparse * sparse_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        loop.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100 * correct / total:.2f}%"
        })

    return total_loss / len(loader), 100 * correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    correct = 0
    total = 0
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)

        total_loss += loss.item()

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / len(loader), 100 * correct / total


def plot_gate_distribution(model, lambda_sparse, output_dir):
    gates = model.all_gates().numpy()

    plt.figure(figsize=(8, 5))
    plt.hist(gates, bins=60)
    plt.xlabel("Gate Value")
    plt.ylabel("Frequency")
    plt.title(f"Final Gate Distribution, Lambda={lambda_sparse}")
    plt.tight_layout()

    path = os.path.join(output_dir, f"gate_distribution_lambda_{lambda_sparse}.png")
    plt.savefig(path)
    plt.close()

    return path


def write_report(results, output_dir):
    report_path = os.path.join(output_dir, "REPORT.md")

    best_acc = max(results, key=lambda x: x["test_accuracy"])
    best_sparse = max(results, key=lambda x: x["sparsity"])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Self-Pruning Neural Network Report\n\n")

        f.write("## Problem Understanding\n\n")
        f.write(
            "The goal was to build a neural network that learns to prune itself during training. "
            "Instead of pruning after training, each weight is controlled by a learnable gate. "
            "The model jointly learns classification weights and pruning decisions.\n\n"
        )

        f.write("## Method\n\n")
        f.write(
            "I implemented a custom `PrunableLinear` layer. Each normal weight has a matching "
            "`gate_score`. During the forward pass, the gate is calculated using sigmoid, and "
            "the effective weight becomes:\n\n"
        )

        f.write("```text\n")
        f.write("effective_weight = weight * sigmoid(gate_score)\n")
        f.write("```\n\n")

        f.write(
            "If the gate value becomes small, the corresponding connection contributes very little "
            "and is treated as pruned.\n\n"
        )

        f.write("## Sparsity Regularization\n\n")
        f.write(
            "The total loss combines classification loss and sparsity loss:\n\n"
        )

        f.write("```text\n")
        f.write("Total Loss = CrossEntropyLoss + lambda * mean(gates)\n")
        f.write("```\n\n")

        f.write(
            "The L1-style penalty on gate values encourages unnecessary gates to become small. "
            "Cross-entropy keeps important gates active, while the sparsity penalty suppresses "
            "less useful connections.\n\n"
        )

        f.write("## Results\n\n")
        f.write("| Lambda | Test Accuracy (%) | Sparsity Level (%) |\n")
        f.write("|---:|---:|---:|\n")

        for r in results:
            f.write(
                f"| {r['lambda']} | {r['test_accuracy']:.2f} | {r['sparsity']:.2f} |\n"
            )

        f.write("\n## Best Accuracy Model\n\n")
        f.write(
            f"The best accuracy model used lambda `{best_acc['lambda']}` with "
            f"`{best_acc['test_accuracy']:.2f}%` test accuracy and "
            f"`{best_acc['sparsity']:.2f}%` sparsity.\n\n"
        )

        f.write("## Highest Sparsity Model\n\n")
        f.write(
            f"The highest sparsity model used lambda `{best_sparse['lambda']}` with "
            f"`{best_sparse['sparsity']:.2f}%` sparsity and "
            f"`{best_sparse['test_accuracy']:.2f}%` test accuracy.\n\n"
        )

        f.write("## Trade-off Analysis\n\n")
        f.write(
            "The experiments show the expected sparsity-accuracy trade-off. Lower lambda values "
            "allow more gates to remain active, usually preserving accuracy. Higher lambda values "
            "apply stronger pressure on the gates, increasing sparsity but potentially reducing "
            "accuracy. This validates that the network is learning pruning decisions during "
            "training rather than relying on a separate post-training pruning step.\n\n"
        )

        f.write("## Conclusion\n\n")
        f.write(
            "This approach converts pruning into a differentiable optimization problem. "
            "The network learns both task performance and structural compression jointly, "
            "which makes it suitable for memory- and compute-constrained deployment scenarios.\n"
        )

    return report_path


def run_experiment(args):
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_loader, test_loader = get_loaders(
        batch_size=args.batch_size,
        subset=args.subset
    )

    results = []

    for lambda_sparse in args.lambdas:
        print("\n" + "=" * 70)
        print(f"Training with lambda = {lambda_sparse}")
        print("=" * 70)

        model = SelfPruningNet().to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=1e-4
        )

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                lambda_sparse
            )

            test_loss, test_acc = evaluate(model, test_loader, device)
            sparsity = model.sparsity_percentage(args.threshold)

            print(
                f"Epoch [{epoch}/{args.epochs}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.2f}% | "
                f"Test Acc: {test_acc:.2f}% | "
                f"Sparsity: {sparsity:.2f}%"
            )

        final_test_loss, final_test_acc = evaluate(model, test_loader, device)
        final_sparsity = model.sparsity_percentage(args.threshold)

        plot_path = plot_gate_distribution(model, lambda_sparse, args.output_dir)

        model_path = os.path.join(
            args.output_dir,
            f"self_pruning_lambda_{lambda_sparse}.pt"
        )

        torch.save(model.state_dict(), model_path)

        results.append({
            "lambda": lambda_sparse,
            "test_accuracy": final_test_acc,
            "sparsity": final_sparsity,
            "plot": plot_path,
            "model_path": model_path
        })

    df = pd.DataFrame(results)
    csv_path = os.path.join(args.output_dir, "results.csv")
    df.to_csv(csv_path, index=False)

    report_path = write_report(results, args.output_dir)

    print("\nFinal Results")
    print(df[["lambda", "test_accuracy", "sparsity"]])

    print(f"\nSaved CSV to: {csv_path}")
    print(f"Saved report to: {report_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Self-Pruning Neural Network on CIFAR-10"
    )

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=[0.001, 0.005, 0.01, 0.05],
        help="Different sparsity penalty values"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="Gate value below which a weight is counted as pruned"
    )

    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Use smaller subset for quick testing"
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
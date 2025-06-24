# Build your model (example with LSTM)
#model = Sequential([
    # Example:
    # Embedding(input_dim=10000, output_dim=128, input_length=X_train.shape[1]),
    # LSTM(64, return_sequences=True),
    # Dropout(0.5),
    # Dense(1, activation='sigmoid')
#])
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class AstronomyCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(AstronomyCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 28 * 28, 512)  # Assuming input size is 224x224
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)  # Output for 2 classes

        # Fully connected layers (adjusted for 128x128 input)
        #self.fc1 = nn.Linear(128 * 16 * 16, 512)  # Updated to match reduced input size
        #self.fc2 = nn.Linear(512, 128)
        #self.fc3 = nn.Linear(128, num_classes)  # Output for 2 classes


        # Dropout for regularization
        self.dropout = nn.Dropout(0.6)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)  # Flattening

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x




# Compile the model
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AstronomyCNN(num_classes=2).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)




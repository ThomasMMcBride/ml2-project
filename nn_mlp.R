# 2-Hidden Layer MLP

# Activation functions
relu <- function(x) pmax(0, x)
relu_derivative <- function(x) ifelse(x > 0, 1, 0)

sigmoid <- function(x) 1 / (1 + exp(-x))
sigmoid_derivative <- function(x) {
  s <- sigmoid(x)
  s * (1 - s)
}

# Initialize weights
initialize_parameters <- function(input_size, hidden1_size, hidden2_size, output_size) {
  set.seed(42)
  list(
    W1 = matrix(rnorm(hidden1_size * input_size, sd = 0.01), nrow = hidden1_size),
    b1 = matrix(0, nrow = hidden1_size, ncol = 1),
    W2 = matrix(rnorm(hidden2_size * hidden1_size, sd = 0.01), nrow = hidden2_size),
    b2 = matrix(0, nrow = hidden2_size, ncol = 1),
    W3 = matrix(rnorm(output_size * hidden2_size, sd = 0.01), nrow = output_size),
    b3 = matrix(0, nrow = output_size, ncol = 1)
  )
}

# Forward pass (single sample)
forward_propagation <- function(X, params) {
  Z1 <- params$W1 %*% X + params$b1
  A1 <- relu(Z1)

  Z2 <- params$W2 %*% A1 + params$b2
  A2 <- relu(Z2)

  Z3 <- params$W3 %*% A2 + params$b3
  A3 <- sigmoid(Z3)

  list(Z1=Z1, A1=A1, Z2=Z2, A2=A2, Z3=Z3, A3=A3)
}

# Backward pass (single sample)
backward_propagation <- function(X, Y, params, cache) {
  dZ3 <- (cache$A3 - Y) * sigmoid_derivative(cache$Z3)
  dW3 <- dZ3 %*% t(cache$A2)
  db3 <- dZ3

  dA2 <- t(params$W3) %*% dZ3
  dZ2 <- dA2 * relu_derivative(cache$Z2)
  dW2 <- dZ2 %*% t(cache$A1)
  db2 <- dZ2

  dA1 <- t(params$W2) %*% dZ2
  dZ1 <- dA1 * relu_derivative(cache$Z1)
  dW1 <- dZ1 %*% t(X)
  db1 <- dZ1

  list(
    dW1=dW1, db1=db1,
    dW2=dW2, db2=db2,
    dW3=dW3, db3=db3
  )
}

# Update parameters
update_parameters <- function(params, grads, lr) {
  params$W1 <- params$W1 - lr * grads$dW1
  params$b1 <- params$b1 - lr * grads$db1
  params$W2 <- params$W2 - lr * grads$dW2
  params$b2 <- params$b2 - lr * grads$db2
  params$W3 <- params$W3 - lr * grads$dW3
  params$b3 <- params$b3 - lr * grads$db3
  params
}

# Training loop (1 sample at a time)
train_nn <- function(X_all, Y_all, iterations = 10, learning_rate = 0.01) {
  input_size <- nrow(X_all)
  hidden_size <- 100
  output_size <- 1

  params <- initialize_parameters(input_size, hidden_size, hidden_size, output_size)
  n_samples <- ncol(X_all)

  for (iter in 1:iterations) {
    total_loss <- 0
    for (i in 1:n_samples) {
      X <- matrix(X_all[, i], ncol = 1)
      Y <- matrix(Y_all[i], nrow = 1)

      cache <- forward_propagation(X, params)
      loss <- mean((cache$A3 - Y)^2)
      total_loss <- total_loss + loss

      grads <- backward_propagation(X, Y, params, cache)
      params <- update_parameters(params, grads, learning_rate)
    }
    cat("Iter:", iter, "Loss:", round(total_loss / n_samples, 6), "\n")
  }

  params
}

# Predict on multiple samples
predict_nn <- function(X, params) {
  apply(X, 2, function(x) {
    x <- matrix(x, ncol = 1)
    A1 <- relu(params$W1 %*% x + params$b1)
    A2 <- relu(params$W2 %*% A1 + params$b2)
    A3 <- sigmoid(params$W3 %*% A2 + params$b3)
    A3
  })
}

data <- read.csv("layoffs_cleaned.csv")

# Extract features and target
X_data <- t(as.matrix(data[, 3:69]))           # 67 x 1
Y_data <- data$percentage_laid_off             # N-length vector

# Normalize inputs to [0,1] (feature-wise)
normalize <- function(x) {
  rng <- range(x, na.rm = TRUE)
  if (rng[1] == rng[2]) return(rep(0, length(x)))
  (x - rng[1]) / (rng[2] - rng[1])
}
X_data <- apply(X_data, 1, normalize)
X_data <- t(X_data)

model <- train_nn(X_data, Y_data, iterations = 100, learning_rate = 0.05)

preds <- predict_nn(X_data, model)

cat("\nPredictions vs Actuals:\n")
print(data.frame(Predicted = round(preds[1:10], 3), Actual = round(Y_data[1:10], 3)))

# R2 score
r2_score <- function(y_true, y_pred) {
  ss_res <- sum((y_true - y_pred)^2)
  ss_tot <- sum((y_true - mean(y_true))^2)
  1 - (ss_res / ss_tot)
}
print(paste("R2 Score:", round(r2_score(Y_data, preds), 4)))

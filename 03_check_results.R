# 03_check_results.R -----------------------------------------------------
# 1. Parameter recovery: do 90% credible intervals contain the true values?
# 2. Occupancy model vs naive model: estimated proportion of occupied sites
#    and the effect of forest cover.
# 3. Posterior predictive check: number of sites with >= 1 detection.

source("R/00_setup.R")
sim       <- readRDS("data/sim_data.rds")
fit_occ   <- readRDS("output/fit_occupancy.rds")
fit_naive <- readRDS("output/fit_naive.rds")

q90 <- function(x) quantile(x, c(0.05, 0.95))

# ---- 1. Parameter recovery ---------------------------------------------
pars  <- c("beta[1]", "beta[2]", "alpha[1]", "alpha[2]", "alpha[3]")
labels <- c("psi: intercept", "psi: forest",
            "p: intercept", "p: forest", "p: effort")
truth <- c(sim$truth$beta, sim$truth$alpha)

summ <- fit_occ$summary(pars, mean, q90, "rhat", "ess_bulk")
recovery <- data.frame(
  parameter = labels, true = truth,
  mean = round(summ$mean, 2), q5 = round(summ$`5%`, 2), q95 = round(summ$`95%`, 2),
  covered = truth >= summ$`5%` & truth <= summ$`95%`,
  rhat = round(summ$rhat, 3), ess_bulk = round(summ$ess_bulk)
)
print(recovery, row.names = FALSE)
write.csv(recovery, "output/parameter_recovery.csv", row.names = FALSE)

png("output/figures/parameter_recovery.png", width = 1600, height = 1000, res = 200)
op <- par(mar = c(4, 9, 2, 1))
plot(recovery$mean, 5:1, xlim = range(c(recovery$q5, recovery$q95, truth)) + c(-0.2, 0.2),
     pch = 19, yaxt = "n", ylab = "", xlab = "Logit scale",
     main = "Posterior mean and 90% interval vs true value")
segments(recovery$q5, 5:1, recovery$q95, 5:1, lwd = 2)
points(truth, 5:1, pch = 4, col = "red", cex = 1.6, lwd = 2)
axis(2, at = 5:1, labels = labels, las = 1)
abline(v = 0, lty = 3, col = "grey50")
legend("bottomright", c("Posterior mean (90% CrI)", "True value"),
       pch = c(19, 4), col = c("black", "red"), bty = "n")
par(op); dev.off()

# ---- 2. Occupancy model vs naive model ---------------------------------
d_occ   <- as_draws_df(fit_occ$draws(c("psi_fs", "beta[2]")))
d_naive <- as_draws_df(fit_naive$draws(c("psi_naive", "beta[2]")))

compare <- data.frame(
  quantity = c("Proportion of sites occupied", "Forest effect on occupancy (logit)"),
  true = round(c(sim$truth$psi_fs, sim$truth$beta[2]), 2),
  occupancy_model = c(sprintf("%.2f [%.2f, %.2f]", mean(d_occ$psi_fs), q90(d_occ$psi_fs)[1], q90(d_occ$psi_fs)[2]),
                      sprintf("%.2f [%.2f, %.2f]", mean(d_occ$`beta[2]`), q90(d_occ$`beta[2]`)[1], q90(d_occ$`beta[2]`)[2])),
  naive_model = c(sprintf("%.2f [%.2f, %.2f]", mean(d_naive$psi_naive), q90(d_naive$psi_naive)[1], q90(d_naive$psi_naive)[2]),
                  sprintf("%.2f [%.2f, %.2f]", mean(d_naive$`beta[2]`), q90(d_naive$`beta[2]`)[1], q90(d_naive$`beta[2]`)[2]))
)
print(compare, row.names = FALSE)
write.csv(compare, "output/occupancy_vs_naive.csv", row.names = FALSE)

overlay_hist <- function(a, b, truth, main, xlab = "") {
  br <- seq(min(c(a, b, truth)), max(c(a, b, truth)), length.out = 50)
  ha <- hist(a, breaks = br, plot = FALSE); hb <- hist(b, breaks = br, plot = FALSE)
  plot(ha, freq = FALSE, col = adjustcolor("steelblue", 0.6), border = NA,
       ylim = c(0, max(ha$density, hb$density)), main = main, xlab = xlab)
  plot(hb, freq = FALSE, col = adjustcolor("orange", 0.6), border = NA, add = TRUE)
  abline(v = truth, col = "red", lwd = 2)
}

png("output/figures/occupancy_vs_naive.png", width = 1600, height = 900, res = 200)
op <- par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))
overlay_hist(d_occ$psi_fs, d_naive$psi_naive, sim$truth$psi_fs, "Proportion of sites occupied")
legend("topright", c("Occupancy model", "Naive model", "Truth"),
       fill = c(adjustcolor("steelblue", 0.6), adjustcolor("orange", 0.6), NA),
       border = NA, lty = c(NA, NA, 1), col = c(NA, NA, "red"), bty = "n", cex = 0.75)
overlay_hist(d_occ$`beta[2]`, d_naive$`beta[2]`, sim$truth$beta[2],
             "Forest effect on occupancy", "logit scale")
par(op); dev.off()

# ---- 3. Posterior predictive check -------------------------------------
n_det_obs <- sum(rowSums(sim$stan_data$y) > 0)
n_det_rep <- as.vector(fit_occ$draws("n_detected_rep", format = "matrix"))
ppp <- mean(n_det_rep >= n_det_obs)
cat("\nPosterior predictive check: sites with >= 1 detection\n")
cat("  observed =", n_det_obs, "; replicated 90% interval =",
    paste(q90(n_det_rep), collapse = " to "),
    "; posterior predictive p-value =", round(ppp, 2), "\n")

png("output/figures/ppc_detected_sites.png", width = 1400, height = 900, res = 200)
hist(n_det_rep, breaks = 25, col = "grey80", border = "white",
     main = "Posterior predictive check", xlab = "Number of sites with >= 1 detection")
abline(v = n_det_obs, col = "red", lwd = 2)
legend("topright", c("Replicated data", "Observed"), fill = c("grey80", NA),
       border = NA, lty = c(NA, 1), col = c(NA, "red"), bty = "n")
dev.off()

cat("\nFigures written to output/figures/\n")

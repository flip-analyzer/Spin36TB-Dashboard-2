import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
import warnings
from datetime import datetime, timedelta


class MomentumBacktester:
    """
    Backtesting framework for momentum trading strategies following López de Prado's methodology.
    
    Implements realistic constraints including:
    - Transaction costs
    - Market impact
    - Position sizing
    - Risk management
    - Realistic execution delays
    """
    
    def __init__(self,
                 initial_capital: float = 100000,
                 transaction_cost: float = 0.001,
                 market_impact: float = 0.0005,
                 max_position_size: float = 0.1,
                 risk_free_rate: float = 0.02,
                 execution_delay: int = 1):
        """
        Parameters:
        -----------
        initial_capital : float
            Starting capital
        transaction_cost : float
            Transaction cost as percentage of trade value
        market_impact : float
            Market impact cost as percentage of trade value
        max_position_size : float
            Maximum position size as fraction of portfolio
        risk_free_rate : float
            Annual risk-free rate
        execution_delay : int
            Number of periods delay for execution
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.market_impact = market_impact
        self.max_position_size = max_position_size
        self.risk_free_rate = risk_free_rate
        self.execution_delay = execution_delay
        
        # Track portfolio state
        self.portfolio_value = initial_capital
        self.positions = {}
        self.cash = initial_capital
        self.trades = []
        self.portfolio_history = []
        
    def backtest(self,
                 signals: pd.Series,
                 prices: pd.DataFrame,
                 probabilities: Optional[pd.Series] = None,
                 position_sizing_method: str = 'kelly',
                 rebalance_frequency: str = 'D') -> Dict:
        """
        Run backtest with given signals
        
        Parameters:
        -----------
        signals : pd.Series
            Trading signals (-1, 0, 1)
        prices : pd.DataFrame
            Price data (must include 'Close')
        probabilities : pd.Series, optional
            Prediction probabilities for position sizing
        position_sizing_method : str
            Method for position sizing ('fixed', 'kelly', 'volatility')
        rebalance_frequency : str
            Frequency of rebalancing ('D', 'W', 'M')
            
        Returns:
        --------
        dict
            Backtest results
        """
        # Reset portfolio state
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.portfolio_history = []
        
        # Align data
        common_index = signals.index.intersection(prices.index)
        signals = signals.loc[common_index]
        prices = prices.loc[common_index]
        
        if probabilities is not None:
            probabilities = probabilities.loc[common_index]
        
        # Calculate returns for volatility-based sizing
        returns = prices['Close'].pct_change().dropna()
        volatility = returns.rolling(21).std() * np.sqrt(252)
        
        # Run backtest
        for i, (date, signal) in enumerate(signals.items()):
            if i < self.execution_delay:
                continue
                
            current_price = prices.loc[date, 'Close']
            
            # Get execution price (delayed)
            exec_date = signals.index[i - self.execution_delay]
            exec_price = prices.loc[exec_date, 'Close']
            
            # Calculate position size
            position_size = self._calculate_position_size(
                signal,
                probabilities.loc[date] if probabilities is not None else None,
                volatility.loc[date] if date in volatility.index else 0.02,
                position_sizing_method
            )
            
            # Execute trade if signal changed
            current_position = self.positions.get('position', 0)
            target_position = signal * position_size
            
            if abs(target_position - current_position) > 0.01:  # Minimum trade threshold
                self._execute_trade(date, exec_price, current_position, target_position)
            
            # Update portfolio value
            self._update_portfolio_value(date, current_price)
            
        # Calculate performance metrics
        return self._calculate_performance_metrics(prices)
    
    def _calculate_position_size(self,
                                 signal: float,
                                 probability: Optional[float],
                                 volatility: float,
                                 method: str) -> float:
        """Calculate position size based on method"""
        if signal == 0:
            return 0.0
            
        if method == 'fixed':
            return self.max_position_size
            
        elif method == 'kelly':
            if probability is None:
                return self.max_position_size * 0.5
                
            # Kelly criterion
            p = probability  # Probability of success
            b = 1.0  # Payoff ratio (assuming 1:1)
            f = (p * b - (1 - p)) / b
            
            # Apply safety factor and cap at max position size
            kelly_fraction = max(0, min(f * 0.25, self.max_position_size))
            return kelly_fraction
            
        elif method == 'volatility':
            # Inverse volatility sizing
            if volatility <= 0:
                return 0.0
                
            target_volatility = 0.15  # 15% annual volatility target
            vol_adjustment = min(target_volatility / volatility, 2.0)  # Cap at 2x
            
            base_size = self.max_position_size * vol_adjustment
            return min(base_size, self.max_position_size)
            
        else:
            return self.max_position_size
    
    def _execute_trade(self,
                       date: pd.Timestamp,
                       price: float,
                       current_position: float,
                       target_position: float) -> None:
        """Execute a trade with realistic costs"""
        trade_size = target_position - current_position
        trade_value = abs(trade_size) * self.portfolio_value * price
        
        # Calculate costs
        transaction_costs = trade_value * self.transaction_cost
        market_impact_costs = trade_value * self.market_impact * abs(trade_size)
        total_costs = transaction_costs + market_impact_costs
        
        # Check if we have enough cash for the trade
        required_cash = abs(trade_size) * self.portfolio_value * price + total_costs
        
        if trade_size > 0 and required_cash > self.cash:
            # Reduce trade size if insufficient cash
            max_trade = (self.cash - total_costs) / (self.portfolio_value * price)
            trade_size = min(trade_size, max_trade)
            target_position = current_position + trade_size
        
        if abs(trade_size) > 0.001:  # Minimum trade size
            # Record trade
            trade = {
                'date': date,
                'price': price,
                'trade_size': trade_size,
                'current_position': current_position,
                'target_position': target_position,
                'transaction_costs': transaction_costs,
                'market_impact_costs': market_impact_costs,
                'total_costs': total_costs
            }
            self.trades.append(trade)
            
            # Update positions and cash
            self.positions['position'] = target_position
            self.cash -= total_costs
    
    def _update_portfolio_value(self, date: pd.Timestamp, price: float) -> None:
        """Update portfolio value"""
        position = self.positions.get('position', 0)
        position_value = position * self.portfolio_value * price
        self.portfolio_value = self.cash + position_value
        
        # Record portfolio history
        self.portfolio_history.append({
            'date': date,
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'position': position,
            'position_value': position_value
        })
    
    def _calculate_performance_metrics(self, prices: pd.DataFrame) -> Dict:
        """Calculate comprehensive performance metrics"""
        portfolio_df = pd.DataFrame(self.portfolio_history).set_index('date')
        
        if len(portfolio_df) == 0:
            return {'error': 'No portfolio history available'}
        
        # Portfolio returns
        portfolio_returns = portfolio_df['portfolio_value'].pct_change().dropna()
        
        # Benchmark returns (buy and hold)
        price_returns = prices['Close'].pct_change().dropna()
        benchmark_returns = price_returns.loc[portfolio_returns.index]
        
        # Basic metrics
        total_return = (portfolio_df['portfolio_value'].iloc[-1] / self.initial_capital) - 1
        annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - self.risk_free_rate) / volatility if volatility > 0 else 0
        
        # Drawdown analysis
        cumulative_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Win/Loss analysis
        trades_df = pd.DataFrame(self.trades)
        if len(trades_df) > 0:
            # Calculate trade P&L
            trade_pnl = []
            for i in range(len(trades_df) - 1):
                entry_price = trades_df.iloc[i]['price']
                exit_price = trades_df.iloc[i + 1]['price']
                position = trades_df.iloc[i]['target_position']
                pnl = position * (exit_price - entry_price) * self.initial_capital
                trade_pnl.append(pnl)
            
            if trade_pnl:
                winning_trades = [pnl for pnl in trade_pnl if pnl > 0]
                losing_trades = [pnl for pnl in trade_pnl if pnl < 0]
                
                win_rate = len(winning_trades) / len(trade_pnl) if trade_pnl else 0
                avg_win = np.mean(winning_trades) if winning_trades else 0
                avg_loss = np.mean(losing_trades) if losing_trades else 0
                profit_factor = abs(sum(winning_trades) / sum(losing_trades)) if losing_trades else np.inf
            else:
                win_rate = avg_win = avg_loss = profit_factor = 0
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0
        
        # Risk metrics
        sortino_ratio = self._calculate_sortino_ratio(portfolio_returns)
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else np.inf
        
        # Information ratio vs benchmark
        tracking_error = (portfolio_returns - benchmark_returns).std() * np.sqrt(252)
        information_ratio = ((annualized_return - benchmark_returns.mean() * 252) / tracking_error 
                           if tracking_error > 0 else 0)
        
        # Transaction cost analysis
        total_costs = sum([trade['total_costs'] for trade in self.trades])
        cost_drag = total_costs / self.initial_capital
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'information_ratio': information_ratio,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'num_trades': len(trades_df),
            'total_costs': total_costs,
            'cost_drag': cost_drag,
            'final_portfolio_value': portfolio_df['portfolio_value'].iloc[-1],
            'benchmark_return': (benchmark_returns.mean() * 252) if len(benchmark_returns) > 0 else 0
        }
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio"""
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return np.inf
            
        downside_deviation = downside_returns.std() * np.sqrt(252)
        excess_return = returns.mean() * 252 - self.risk_free_rate
        
        return excess_return / downside_deviation if downside_deviation > 0 else 0
    
    def get_portfolio_history(self) -> pd.DataFrame:
        """Get portfolio history as DataFrame"""
        return pd.DataFrame(self.portfolio_history).set_index('date')
    
    def get_trades(self) -> pd.DataFrame:
        """Get trades as DataFrame"""
        return pd.DataFrame(self.trades)
    
    def plot_results(self, prices: pd.DataFrame = None) -> None:
        """Plot backtest results"""
        try:
            import matplotlib.pyplot as plt
            
            portfolio_df = self.get_portfolio_history()
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Portfolio value over time
            axes[0, 0].plot(portfolio_df.index, portfolio_df['portfolio_value'])
            axes[0, 0].set_title('Portfolio Value Over Time')
            axes[0, 0].set_ylabel('Portfolio Value')
            
            # Drawdown
            returns = portfolio_df['portfolio_value'].pct_change()
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            
            axes[0, 1].fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
            axes[0, 1].set_title('Drawdown')
            axes[0, 1].set_ylabel('Drawdown')
            
            # Returns distribution
            axes[1, 0].hist(returns.dropna(), bins=50, alpha=0.7)
            axes[1, 0].set_title('Returns Distribution')
            axes[1, 0].set_xlabel('Returns')
            axes[1, 0].set_ylabel('Frequency')
            
            # Position over time
            axes[1, 1].plot(portfolio_df.index, portfolio_df['position'])
            axes[1, 1].set_title('Position Over Time')
            axes[1, 1].set_ylabel('Position')
            axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("matplotlib not available for plotting")


class WalkForwardBacktester:
    """
    Walk-forward backtesting for out-of-sample validation
    """
    
    def __init__(self, 
                 backtester: MomentumBacktester,
                 train_periods: int = 252,
                 test_periods: int = 63,
                 step_size: int = 21):
        """
        Parameters:
        -----------
        backtester : MomentumBacktester
            Base backtester instance
        train_periods : int
            Training window size
        test_periods : int
            Test window size
        step_size : int
            Step size for moving window
        """
        self.backtester = backtester
        self.train_periods = train_periods
        self.test_periods = test_periods
        self.step_size = step_size
        
    def run_walk_forward(self,
                         model,
                         features: pd.DataFrame,
                         labels: pd.Series,
                         prices: pd.DataFrame) -> Dict:
        """
        Run walk-forward backtesting
        
        Parameters:
        -----------
        model : MomentumMLModel
            ML model to use
        features : pd.DataFrame
            Feature matrix
        labels : pd.Series
            Target labels
        prices : pd.DataFrame
            Price data
            
        Returns:
        --------
        dict
            Walk-forward results
        """
        results = []
        n_obs = len(features)
        
        for i in range(self.train_periods, n_obs - self.test_periods, self.step_size):
            # Define windows
            train_start = max(0, i - self.train_periods)
            train_end = i
            test_start = i
            test_end = min(i + self.test_periods, n_obs)
            
            # Prepare data
            X_train = features.iloc[train_start:train_end]
            y_train = labels.iloc[train_start:train_end]
            X_test = features.iloc[test_start:test_end]
            y_test = labels.iloc[test_start:test_end]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Generate signals
            signals = pd.Series(model.predict(X_test), index=X_test.index)
            probabilities = pd.Series(
                model.predict_proba(X_test)[:, 1],  # Probability of positive class
                index=X_test.index
            )
            
            # Run backtest on test period
            test_prices = prices.iloc[test_start:test_end]
            backtest_results = self.backtester.backtest(
                signals, test_prices, probabilities
            )
            
            # Store results
            period_result = {
                'period': len(results),
                'train_start': X_train.index[0],
                'train_end': X_train.index[-1],
                'test_start': X_test.index[0],
                'test_end': X_test.index[-1],
                'train_size': len(X_train),
                'test_size': len(X_test)
            }
            period_result.update(backtest_results)
            results.append(period_result)
        
        # Aggregate results
        return self._aggregate_walk_forward_results(results)
    
    def _aggregate_walk_forward_results(self, results: List[Dict]) -> Dict:
        """Aggregate walk-forward results"""
        if not results:
            return {}
            
        # Convert to DataFrame for easier analysis
        results_df = pd.DataFrame(results)
        
        # Calculate aggregate metrics
        total_return = results_df['total_return'].mean()
        volatility = results_df['volatility'].mean()
        sharpe_ratio = results_df['sharpe_ratio'].mean()
        max_drawdown = results_df['max_drawdown'].min()  # Worst drawdown
        win_rate = results_df['win_rate'].mean()
        
        # Consistency metrics
        positive_periods = (results_df['total_return'] > 0).mean()
        return_std = results_df['total_return'].std()
        
        return {
            'periods_tested': len(results),
            'avg_total_return': total_return,
            'avg_volatility': volatility,
            'avg_sharpe_ratio': sharpe_ratio,
            'worst_max_drawdown': max_drawdown,
            'avg_win_rate': win_rate,
            'positive_periods_pct': positive_periods,
            'return_consistency': 1 / (return_std + 1e-8),  # Higher is more consistent
            'detailed_results': results_df
        }
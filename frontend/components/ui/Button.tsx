'use client'

import React from 'react'
import { motion } from 'framer-motion'
import { Loader2 } from 'lucide-react'
import { cn, type MotionSafe } from '@/lib/utils'

export interface ButtonProps extends MotionSafe<React.ButtonHTMLAttributes<HTMLButtonElement>> {
  variant?: 'primary' | 'secondary' | 'ghost' | 'success' | 'outline'
  size?: 'sm' | 'md' | 'lg'
  loading?: boolean
  icon?: React.ReactNode
  iconPosition?: 'left' | 'right'
  fullWidth?: boolean
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      className,
      variant = 'primary',
      size = 'md',
      loading = false,
      icon,
      iconPosition = 'left',
      fullWidth = false,
      children,
      disabled,
      ...props
    },
    ref
  ) => {
    const baseClasses = 'inline-flex items-center justify-center gap-2 font-medium transition-all duration-200 ease-out focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed disabled:pointer-events-none'

    const variants = {
      primary: 'bg-harvest-500 text-stone-900 font-semibold shadow-soft hover:bg-harvest-600 hover:shadow-soft-lg focus:ring-harvest-500/20',
      secondary: 'bg-white text-stone-700 border-2 border-stone-200 shadow-soft hover:bg-stone-50 hover:border-stone-300 hover:shadow-soft-lg focus:ring-stone-500/20',
      ghost: 'bg-transparent text-stone-700 hover:bg-stone-100 active:bg-stone-200 focus:ring-stone-500/20',
      success: 'bg-field-600 text-stone-900 font-semibold shadow-soft hover:bg-field-700 hover:shadow-soft-lg focus:ring-field-600/20',
      outline: 'border-2 border-harvest-500 text-harvest-600 hover:bg-harvest-50 focus:ring-harvest-500/20',
    }

    const sizes = {
      sm: 'px-4 py-2 text-sm rounded-lg',
      md: 'px-6 py-3 text-sm rounded-xl',
      lg: 'px-8 py-4 text-base rounded-xl',
    }

    return (
      <motion.button
        ref={ref}
        className={cn(
          baseClasses,
          variants[variant],
          sizes[size],
          fullWidth && 'w-full',
          className
        )}
        disabled={disabled || loading}
        whileHover={{ y: -2, scale: 1.01 }}
        whileTap={{ y: 0, scale: 0.98 }}
        transition={{ type: 'spring', stiffness: 400, damping: 25 }}
        {...props}
      >
        {loading && <Loader2 className="w-4 h-4 animate-spin" />}
        {!loading && icon && iconPosition === 'left' && icon}
        {children}
        {!loading && icon && iconPosition === 'right' && icon}
      </motion.button>
    )
  }
)

Button.displayName = 'Button'

export default Button

'use client'

import React from 'react'
import { cn } from '@/lib/utils'

export interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: 'success' | 'warning' | 'error' | 'neutral'
  size?: 'sm' | 'md' | 'lg'
  icon?: React.ReactNode
  dot?: boolean
}

const Badge = React.forwardRef<HTMLSpanElement, BadgeProps>(
  ({ className, variant = 'neutral', size = 'md', icon, dot, children, ...props }, ref) => {
    const baseClasses = 'inline-flex items-center gap-1.5 rounded-full font-medium transition-all duration-200'

    const variants = {
      success: 'bg-field-50 text-field-700 border border-field-200',
      warning: 'bg-harvest-50 text-harvest-700 border border-harvest-200',
      error: 'bg-clay-50 text-clay-700 border border-clay-200',
      neutral: 'bg-stone-100 text-stone-700 border border-stone-200',
    }

    const sizes = {
      sm: 'px-2 py-0.5 text-xs',
      md: 'px-3 py-1 text-xs',
      lg: 'px-4 py-1.5 text-sm',
    }

    const dotColors = {
      success: 'bg-field-500',
      warning: 'bg-harvest-500',
      error: 'bg-clay-500',
      neutral: 'bg-stone-500',
    }

    return (
      <span
        ref={ref}
        className={cn(baseClasses, variants[variant], sizes[size], className)}
        {...props}
      >
        {dot && (
          <span
            className={cn('inline-block w-2 h-2 rounded-full animate-pulse-soft', dotColors[variant])}
          />
        )}
        {icon && !dot && icon}
        {children}
      </span>
    )
  }
)

Badge.displayName = 'Badge'

export default Badge

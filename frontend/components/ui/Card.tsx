'use client'

import React from 'react'
import { motion } from 'framer-motion'
import { cn } from '@/lib/utils'

export interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: 'default' | 'hover' | 'glass'
  padding?: 'none' | 'sm' | 'md' | 'lg'
}

const Card = React.forwardRef<HTMLDivElement, CardProps>(
  ({ className, variant = 'default', padding = 'md', children, ...props }, ref) => {
    const baseClasses = 'bg-white rounded-2xl shadow-soft border border-stone-200 transition-all duration-200'

    const variants = {
      default: '',
      hover: 'hover:shadow-soft-lg hover:-translate-y-1 hover:border-stone-300 cursor-pointer',
      glass: 'bg-white/80 backdrop-blur-md border-white/20 shadow-soft-lg',
    }

    const paddings = {
      none: '',
      sm: 'p-4',
      md: 'p-6',
      lg: 'p-8',
    }

    const MotionComponent = variant === 'hover' ? motion.div : 'div'

    return (
      <MotionComponent
        ref={ref as any}
        className={cn(baseClasses, variants[variant], paddings[padding], className)}
        {...(variant === 'hover' && {
          whileHover: { y: -4, scale: 1.01 },
          transition: { type: 'spring', stiffness: 400, damping: 25 },
        })}
        {...props}
      >
        {children}
      </MotionComponent>
    )
  }
)

Card.displayName = 'Card'

export default Card
